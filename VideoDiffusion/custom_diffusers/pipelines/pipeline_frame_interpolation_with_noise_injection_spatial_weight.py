# Adpated from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_stable_video_diffusion.py
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
import copy
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import UNetSpatioTemporalConditionModel
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
        _append_dims,
        tensor2vid,
        _resize_with_antialiasing,
        StableVideoDiffusionPipelineOutput
)
from ..schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class FrameInterpolationWithNoiseInjectionPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.ori_unet = copy.deepcopy(unet)
       
    def _encode_image(
        self,
        image: PipelineImageInput,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.FloatTensor:
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(self, latents: torch.FloatTensor, num_frames: int, decode_chunk_size: int = 14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.FloatTensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return False

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # Pose similarity functions
    def angular_distance(self, q_a, q_b):
        q_a = F.normalize(q_a, dim=-1)
        q_b = F.normalize(q_b, dim=-1)
        dot_product = torch.sum(q_a * q_b, dim=-1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angular_dist = 2.0 * torch.acos(torch.abs(dot_product))
        return angular_dist

    def pose_similarity(self, p_current, p_source, p_target, sigma_q=1.0, sigma_t=1.0):
        q_current, t_current = p_current
        q_source, t_source = p_source
        q_target, t_target = p_target

        # Calculate quaternion weights
        w_q_source = torch.exp(-self.angular_distance(q_current, q_source) / sigma_q)
        w_q_target = torch.exp(-self.angular_distance(q_current, q_target) / sigma_q)

        # Calculate translation weights
        w_t_source = torch.exp(-torch.norm(t_current - t_source, dim=-1) / sigma_t)
        w_t_target = torch.exp(-torch.norm(t_current - t_target, dim=-1) / sigma_t)

        # Combine weights
        w_source = w_q_source * w_t_source
        w_target = w_q_target * w_t_target

        # Normalize weights
        w_sum = w_source + w_target
        w_source = w_source / w_sum
        w_target = w_target / w_sum

        return w_source, w_target

    def matrix_to_translation_quaternion(self, matrix):
        # Assumes matrix is of shape (batch_size, 4, 4)
        translation = matrix[:, :3, 3]
        rotation_matrix = matrix[:, :3, :3]
        quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
        return translation, quaternion

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        # Converts a rotation matrix to a quaternion
        # rotation_matrix shape: (batch_size, 3, 3)
        batch_size = rotation_matrix.shape[0]
        quaternion = torch.zeros((batch_size, 4), device=rotation_matrix.device)

        R = rotation_matrix
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        for i in range(batch_size):
            if trace[i] > 0:
                s = torch.sqrt(trace[i] + 1.0) * 2
                qw = 0.25 * s
                qx = (R[i, 2, 1] - R[i, 1, 2]) / s
                qy = (R[i, 0, 2] - R[i, 2, 0]) / s
                qz = (R[i, 1, 0] - R[i, 0, 1]) / s
            else:
                if R[i, 0, 0] > R[i, 1, 1] and R[i, 0, 0] > R[i, 2, 2]:
                    s = torch.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2]) * 2
                    qw = (R[i, 2, 1] - R[i, 1, 2]) / s
                    qx = 0.25 * s
                    qy = (R[i, 0, 1] + R[i, 1, 0]) / s
                    qz = (R[i, 0, 2] + R[i, 2, 0]) / s
                elif R[i, 1, 1] > R[i, 2, 2]:
                    s = torch.sqrt(1.0 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2]) * 2
                    qw = (R[i, 0, 2] - R[i, 2, 0]) / s
                    qx = (R[i, 0, 1] + R[i, 1, 0]) / s
                    qy = 0.25 * s
                    qz = (R[i, 1, 2] + R[i, 2, 1]) / s
                else:
                    s = torch.sqrt(1.0 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1]) * 2
                    qw = (R[i, 1, 0] - R[i, 0, 1]) / s
                    qx = (R[i, 0, 2] + R[i, 2, 0]) / s
                    qy = (R[i, 1, 2] + R[i, 2, 1]) / s
                    qz = 0.25 * s
            quaternion[i] = torch.tensor([qw, qx, qy, qz], device=rotation_matrix.device)
        return quaternion

    @torch.no_grad()
    def multidiffusion_step(self, latents, t, 
                    image1_embeddings, 
                    image2_embeddings, 
                    image1_latents,
                    image2_latents,
                    added_time_ids, 
                    avg_weight
    ):
        # expand the latents if we are doing classifier free guidance
        latents1 = latents
        latents2 = torch.flip(latents, (1,))
        latent_model_input1 = self.scheduler.scale_model_input(latents1, t)
        latent_model_input2 = self.scheduler.scale_model_input(latents2, t)

        # Concatenate image_latents over channels dimension
        latent_model_input1 = torch.cat([latent_model_input1, image1_latents], dim=2)
        latent_model_input2 = torch.cat([latent_model_input2, image2_latents], dim=2)

        # predict the noise residual
        noise_pred1 = self.ori_unet(
            latent_model_input1,
            t,
            encoder_hidden_states=image1_embeddings,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]
        noise_pred2 = self.unet(
            latent_model_input2,
            t,
            encoder_hidden_states=image2_embeddings,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]

        noise_pred2 = torch.flip(noise_pred2, (1,))
        noise_pred = avg_weight * noise_pred1 + (1 - avg_weight) * noise_pred2
        return noise_pred

    @torch.no_grad()
    def __call__(
        self,
        image1: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        image2: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        pose_embeddings: torch.FloatTensor,
        linear_fuser: torch.nn.Module,
        poses_current: torch.Tensor,
        pose_source: torch.Tensor,
        pose_target: torch.Tensor,
        height: int = 256,
        width: int = 384,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        weighted_average: bool = False,
        noise_injection_steps: int = 0,
        noise_injection_ratio: float=0.0,
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            pose_embeddings (`torch.FloatTensor`):
                The pose embeddings computed from Plücker embeddings and processed through the pose encoder.
            linear_fuser (`torch.nn.Module`):
                The linear module used to fuse the pose embeddings with the image latents.
            poses_current (`torch.Tensor`):
                The current poses for each frame (batch_size, num_frames, 4, 4).
            pose_source (`torch.Tensor`):
                The source pose (batch_size, 4, 4).
            pose_target (`torch.Tensor`):
                The target pose (batch_size, 4, 4).
            (Other arguments are as in the original pipeline.)

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image1, height, width)
        self.check_inputs(image2, height, width)

        # 2. Define call parameters
        if isinstance(image1, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image1, list):
            batch_size = len(image1)
        else:
            batch_size = image1.shape[0]
        device = self._execution_device

        # Convert poses to translations and quaternions
        batch_size = image1.shape[0] if isinstance(image1, torch.Tensor) else 1
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames

        # Convert poses_current (batch_size, num_frames, 4, 4)
        batch_size, num_frames = poses_current.shape[:2]
        poses_current_flat = poses_current.view(batch_size * num_frames, 4, 4)
        t_current_flat, q_current_flat = self.matrix_to_translation_quaternion(poses_current_flat)
        t_current = t_current_flat.view(batch_size, num_frames, -1)
        q_current = q_current_flat.view(batch_size, num_frames, -1)

        # Convert pose_source and pose_target (batch_size, 4, 4)
        t_source, q_source = self.matrix_to_translation_quaternion(pose_source)
        t_target, q_target = self.matrix_to_translation_quaternion(pose_target)

        # Expand source and target poses along frames dimension
        t_source = t_source.unsqueeze(1).expand(-1, num_frames, -1)
        q_source = q_source.unsqueeze(1).expand(-1, num_frames, -1)
        t_target = t_target.unsqueeze(1).expand(-1, num_frames, -1)
        q_target = q_target.unsqueeze(1).expand(-1, num_frames, -1)

        # Compute pose similarities and weights
        w_source, w_target = self.pose_similarity(
            (q_current, t_current),
            (q_source, t_source),
            (q_target, t_target),
            sigma_q=1.0,  # Adjust sigma_q and sigma_t as needed
            sigma_t=1.0
        )
        # Reshape weights for broadcasting
        w_source = w_source.view(batch_size * num_videos_per_prompt, num_frames, 1, 1, 1).to(device).half()
        w_target = w_target.view(batch_size * num_videos_per_prompt, num_frames, 1, 1, 1).to(device).half()

        # Guidance scale
        min_guidance_scale_tensor = torch.full_like(w_source, min_guidance_scale)
        max_guidance_scale_tensor = torch.full_like(w_target, max_guidance_scale)
        guidance_scale = w_source * min_guidance_scale_tensor + w_target * max_guidance_scale_tensor

        # Set guidance scale
        self._guidance_scale = guidance_scale

        # 3. Encode input images
        image1_embeddings = self._encode_image(image1, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        image2_embeddings = self._encode_image(image2, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        fps = fps - 1

        # 4. Encode input images using VAE
        image1 = self.image_processor.preprocess(image1, height=height, width=width).to(device)
        image2 = self.image_processor.preprocess(image2, height=height, width=width).to(device)
        noise = randn_tensor(image1.shape, generator=generator, device=image1.device, dtype=image1.dtype)
        image1 = image1 + noise_aug_strength * noise
        image2 = image2 + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        image1_latent = self._encode_vae_image(image1, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        image1_latent = image1_latent.to(image1_embeddings.dtype)
        image1_latents = image1_latent.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        image1_latents += pose_embeddings
        # Permute to bring channel dimension (4) last for the linear layer
        image1_latents = image1_latents.permute(0, 1, 3, 4, 2).contiguous()  # Shape: [batch, depth, height, width, channels]
        # Apply Linear layer across the channel dimension (now last)
        image1_latents = linear_fuser(image1_latents)
        # Permute back to original shape [batch, channels, depth, height, width]
        image1_latents = image1_latents.permute(0, 1, 4, 2, 3).contiguous()

        image2_latent = self._encode_vae_image(image2, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        image2_latent = image2_latent.to(image2_embeddings.dtype)
        image2_latents = image2_latent.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        image2_latents += torch.flip(pose_embeddings, dims=(1,))
        # Permute to bring channel dimension (4) last for the linear layer
        image2_latents = image2_latents.permute(0, 1, 3, 4, 2).contiguous()  # Shape: [batch, depth, height, width, channels]
        # Apply Linear layer across the channel dimension (now last)
        image2_latents = linear_fuser(image2_latents)
        # Permute back to original shape [batch, channels, depth, height, width]
        image2_latents = image2_latents.permute(0, 1, 4, 2, 3).contiguous()

        # Cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image1_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image1_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        self.ori_unet = self.ori_unet.to(device)

        noise_injection_step_threshold = int(num_inference_steps * noise_injection_ratio)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                noise_pred = self.multidiffusion_step(
                    latents, t,
                    image1_embeddings, image2_embeddings,
                    image1_latents, image2_latents, added_time_ids, w_source
                )
                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                if i < noise_injection_step_threshold and noise_injection_steps > 0:
                    sigma_t = self.scheduler.sigmas[self.scheduler.step_index]
                    sigma_tm1 = self.scheduler.sigmas[self.scheduler.step_index + 1]
                    sigma = torch.sqrt(sigma_t ** 2 - sigma_tm1 ** 2)
                    for j in range(noise_injection_steps):
                        noise = randn_tensor(latents.shape, device=latents.device, dtype=latents.dtype)
                        noise = noise * sigma
                        latents = latents + noise
                        noise_pred = self.multidiffusion_step(
                            latents, t,
                            image1_embeddings, image2_embeddings,
                            image1_latents, image2_latents, added_time_ids, w_source
                        )
                        # Compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                self.scheduler._step_index += 1

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # Cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


