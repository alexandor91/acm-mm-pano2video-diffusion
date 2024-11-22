# Adpated from https://github.com/jeanne-wang/svd_keyframe_interpolation
import os
import torch
import argparse
import copy
import numpy as np
from einops import rearrange
from diffusers.utils import load_image, export_to_video
from diffusers import UNetSpatioTemporalConditionModel
from custom_diffusers.pipelines.pipeline_frame_interpolation_with_noise_injection_spatial_weight import FrameInterpolationWithNoiseInjectionPipeline
from custom_diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from attn_ctrl.attention_control import (
    AttentionStore,
    register_temporal_self_attention_control,
    register_temporal_self_attention_flip_control,
)
from camctrl.pose_adaptor import CameraPoseEncoder
import torch.nn.functional as F
from PIL import Image

os.environ['HF_HOME'] = 'cache'
os.environ['HF_HUB_CACHE'] = 'hub'

class Camera(object):
    def __init__(self, entry):
        """
        Parses a single line from the annotation file and initializes camera parameters.

        Args:
            entry (list of float): List of floats parsed from a line in the annotation file.
        """
        # Entry[0] is frame ID (ignored here)
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        # Skip entries[5:7] as they are zeros or unknown
        w2c_mat_flat = entry[7:19]  # Should be 12 elements for 3x4 matrix
        w2c_mat = np.array(w2c_mat_flat).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def ray_condition(K, c2w, H, W, device):
    """
    Computes Plücker embeddings for rays based on camera intrinsics and poses.

    Args:
        K (torch.Tensor): Camera intrinsics tensor of shape [B, V, 4].
        c2w (torch.Tensor): Camera-to-world poses tensor of shape [B, V, 4, 4].
        H (int): Height of the images.
        W (int): Width of the images.
        device (torch.device): Device on which tensors are allocated.

    Returns:
        torch.Tensor: Plücker embeddings of shape [B, V, H, W, 6].
    """
    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, V, H, W, 6)                        # B, V, H, W, 6
    return plucker


def load_camera_parameters(pose_file):
    """
    Loads camera parameters from the pose file.

    Args:
        pose_file (str): Path to the pose file.

    Returns:
        dict: Dictionary mapping frame IDs to Camera objects.
    """
    with open(pose_file, 'r') as f:
        lines = f.readlines()
    # Skip the first line (e.g., header or metadata)
    lines = lines[1:]
    # Build a dictionary mapping frame_id to Camera objects
    pose_dict = {}
    for line in lines:
        tokens = line.strip().split()
        frame_id = int(tokens[0])
        entries = [float(x) for x in tokens]
        cam_param = Camera(entries)
        pose_dict[frame_id] = cam_param
    return pose_dict

def get_frame_id_from_image_path(image_path):
    """
    Extracts the frame ID from the image filename.

    Args:
        image_path (str): Path to the image file.

    Returns:
        int: Frame ID extracted from the filename.
    """
    filename = os.path.basename(image_path)
    frame_id_str = os.path.splitext(filename)[0]
    frame_id = int(frame_id_str)
    return frame_id

def select_camera_poses_between_frames(pose_dict, start_frame_id, end_frame_id, num_frames):
    """
    Selects camera poses between two frame IDs.

    Args:
        pose_dict (dict): Dictionary mapping frame IDs to Camera objects.
        start_frame_id (int): Frame ID of the starting frame.
        end_frame_id (int): Frame ID of the ending frame.
        num_frames (int): Number of frames to select.

    Returns:
        list of Camera: List of Camera objects corresponding to the selected frames.
    """
    # Ensure start_frame_id <= end_frame_id
    if start_frame_id > end_frame_id:
        start_frame_id, end_frame_id = end_frame_id, start_frame_id

    # Get all frame IDs between start and end (inclusive)
    frame_ids = [frame_id for frame_id in sorted(pose_dict.keys()) if start_frame_id <= frame_id <= end_frame_id]

    # Check if we have enough frames
    if len(frame_ids) < num_frames:
        raise ValueError(f"Not enough frames between frame IDs {start_frame_id} and {end_frame_id} to select {num_frames} frames.")

    # Select frames evenly spaced
    indices = np.linspace(0, len(frame_ids) - 1, num_frames, dtype=int)
    selected_frame_ids = [frame_ids[i] for i in indices]

    # Get the corresponding Camera objects
    selected_cameras = [pose_dict[frame_id] for frame_id in selected_frame_ids]

    return selected_cameras

def compute_plucker_embeddings_from_poses(pose_file, start_frame_id, end_frame_id, num_frames, H, W, device):
    """
    Computes Plücker embeddings from the pose file between two frames.

    Args:
        pose_file (str): Path to the pose file.
        start_frame_id (int): Frame ID of the starting frame.
        end_frame_id (int): Frame ID of the ending frame.
        num_frames (int): Number of frames to select.
        H (int): Height of the images.
        W (int): Width of the images.
        device (torch.device): Device on which tensors are allocated.

    Returns:
        torch.Tensor: Plücker embeddings of shape [num_frames, 6, H, W]
    """
    # Load camera parameters
    pose_dict = load_camera_parameters(pose_file)

    # Select camera poses between the two frames
    selected_cameras = select_camera_poses_between_frames(
        pose_dict, start_frame_id, end_frame_id, num_frames
    )

    # Prepare intrinsics and c2w matrices
    intrinsics = np.asarray([
        [cam.fx*W, cam.fy*H, cam.cx*W, cam.cy*H]
        for cam in selected_cameras
    ], dtype=np.float32)
    intrinsics = torch.from_numpy(intrinsics).unsqueeze(0)  # [1, n_frame, 4]

    c2w_poses = np.array([cam.c2w_mat for cam in selected_cameras], dtype=np.float32)
    c2w = torch.from_numpy(c2w_poses).unsqueeze(0)  # [1, n_frame, 4, 4]

    # Compute Plücker embeddings
    plucker_embedding = ray_condition(
        intrinsics.to(device),
        c2w.to(device),
        H,
        W,
        device=device
    )[0]  # [n_frame, H, W, 6]

    # Rearrange dimensions to [n_frame, 6, H, W]
    plucker_embedding = plucker_embedding.permute(0, 3, 1, 2).contiguous()

    return plucker_embedding

def main(args):

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipe = FrameInterpolationWithNoiseInjectionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=noise_scheduler,
        variant="fp16",
        torch_dtype=torch.float16,
        cache_dir="hub"
    )

    state_dict = pipe.unet.state_dict()
    # Loading finetuned UNet
    finetuned_unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.checkpoint_dir,
        subfolder="unet",
        torch_dtype=torch.float16,
        cache_dir="hub"
    )
    assert finetuned_unet.config.num_frames == 14
    finetuned_state_dict = finetuned_unet.state_dict()
    for name, param in finetuned_state_dict.items():
        state_dict[name] = param
    pipe.unet.load_state_dict(state_dict)

    pipe = pipe.to(args.device)

    # Run inference
    generator = torch.Generator(device=args.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    # Load pose_encoder
    pose_encoder = CameraPoseEncoder()
    pose_encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model_2_state_dict.pth")))
    pose_encoder.eval()
    pose_encoder.requires_grad_(False)
    pose_encoder.to(args.device)
    # Initialize channel_reducer
    channel_reducer = torch.nn.Conv2d(in_channels=320, out_channels=4, kernel_size=1)
    channel_reducer.eval()
    channel_reducer.requires_grad_(False)
    channel_reducer.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model_1_state_dict.pth")))
    channel_reducer.to(args.device)

    # Initialize linear_fuser
    linear_fuser = torch.nn.Linear(4, 4)
    linear_fuser.eval()
    linear_fuser.requires_grad_(False)
    linear_fuser.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model_3_state_dict.pth")))
    linear_fuser.to(args.device).half()

    # Load images
    frame1 = load_image(args.frame1_path)
    frame1 = frame1.resize((384, 256))

    frame2 = load_image(args.frame2_path)
    frame2 = frame2.resize((384, 256))

    # Extract frame IDs from image paths
    frame_id1 = get_frame_id_from_image_path(args.frame1_path)
    frame_id2 = get_frame_id_from_image_path(args.frame2_path)

    num_frames = 14  # Or set via args.num_frames if you add it to argparse

    # Compute Plücker embeddings
    device = args.device
    H, W = 256, 384
    plucker_embedding = compute_plucker_embeddings_from_poses(
        pose_file=args.pose_file,
        start_frame_id=frame_id1,
        end_frame_id=frame_id2,
        num_frames=num_frames,
        H=H,
        W=W,
        device=device
    ).unsqueeze(0).permute(0, 2, 1, 3, 4)

    # Process plucker_embedding through pose_encoder
    with torch.no_grad():
        pose_embedding_features = pose_encoder(plucker_embedding)
        raymap_reduced = channel_reducer(pose_embedding_features)

    # Rearrange the features
    bs = 1  # Since batch size is 1 during inference
    pose_embedding_features = rearrange(raymap_reduced, '(b f) c h w -> b f c h w', b=bs)

    # Prepare pose tensors for spatial weighting
    # Prepare poses_current tensor
    pose_dict = load_camera_parameters(args.pose_file)
    selected_cameras = select_camera_poses_between_frames(
        pose_dict,
        start_frame_id=frame_id1,
        end_frame_id=frame_id2,
        num_frames=num_frames
    )
    c2w_matrices = np.array([cam.c2w_mat for cam in selected_cameras], dtype=np.float32)  # Shape: (num_frames, 4, 4)
    poses_current = torch.from_numpy(c2w_matrices).unsqueeze(0)  # Shape: (1, num_frames, 4, 4)

    # Prepare pose_source and pose_target tensors
    pose_source = torch.from_numpy(pose_dict[frame_id1].c2w_mat).unsqueeze(0)  # Shape: (1, 4, 4)
    pose_target = torch.from_numpy(pose_dict[frame_id2].c2w_mat).unsqueeze(0)  # Shape: (1, 4, 4)

    frames = pipe(
        image1=frame1,
        image2=frame2,
        num_inference_steps=args.num_inference_steps,
        num_frames=num_frames,
        pose_embeddings=pose_embedding_features,
        linear_fuser=linear_fuser,
        generator=generator,
        weighted_average=args.weighted_average,
        noise_injection_steps=args.noise_injection_steps,
        noise_injection_ratio=args.noise_injection_ratio,
        poses_current=poses_current.to(args.device),
        pose_source=pose_source.to(args.device),
        pose_target=pose_target.to(args.device),
    ).frames[0]

    if args.out_path.endswith('.gif'):
        frames[0].save(args.out_path, save_all=True, append_images=frames[1:], duration=142, loop=0)
    else:
        export_to_video(frames, args.out_path, fps=7)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument('--frame1_path', type=str, required=True)
    parser.add_argument('--frame2_path', type=str, required=True)
    parser.add_argument('--pose_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--weighted_average', action='store_true')
    parser.add_argument('--noise_injection_steps', type=int, default=0)
    parser.add_argument('--noise_injection_ratio', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    out_dir = os.path.dirname(args.out_path)
    os.makedirs(out_dir, exist_ok=True)
    main(args)

