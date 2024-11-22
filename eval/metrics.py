import os
import numpy as np
import torch
import torch.nn as nn
import lpips
import cv2
from torchvision import models, transforms
from scipy.linalg import sqrtm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tqdm import tqdm
from sklearn.metrics.pairwise import polynomial_kernel
import json
import tensorflow as tf
from PIL import Image, ImageSequence
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model

# Initialize the InceptionV3 model for FID calculation
fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

# Load the LPIPS model
lpips_model = lpips.LPIPS(net='alex')
import cv2

def preprocess_image(image):
    """Preprocess image for InceptionV3 feature extraction."""
    image = image.resize((299, 299))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def extract_features_from_image(image):
    """Extract features from a PIL Image using InceptionV3."""
    image = preprocess_image(image)
    feature = fid_model.predict(image)
    feature = feature.flatten()
    return feature

def compute_is(generated_features):
    """Compute the Inception Score."""
    # Ensure features are 2D array
    if len(generated_features.shape) == 1:
        generated_features = generated_features.reshape(1, -1)

    # Normalize the features to convert them into probabilities
    generated_features = generated_features - np.max(generated_features, axis=1, keepdims=True)  # Avoid overflow
    generated_features = np.exp(generated_features)  # Apply exp
    generated_features /= np.sum(generated_features, axis=1, keepdims=True)  # Normalize to get probabilities

    # Calculate the KL divergence
    kl_div = generated_features * (np.log(generated_features + 1e-16) - np.log(np.mean(generated_features, axis=0) + 1e-16))

    # Compute the Inception Score
    is_score = np.exp(np.mean(np.sum(kl_div, axis=1)))
    return is_score

def compute_lpips(img1, img2, size=(256, 256)):
    # Resize images
    img1 = img1.resize(size, Image.BICUBIC)
    img2 = img2.resize(size, Image.BICUBIC)
    # Convert to tensors
    img1 = transforms.ToTensor()(img1).unsqueeze(0)
    img2 = transforms.ToTensor()(img2).unsqueeze(0)
    return lpips_model(img1, img2).item()


def compute_psnr(img1, img2):
    """Compute PSNR between two images."""
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    return psnr(img1, img2, data_range=255)

def compute_ssim(img1, img2, win_size=3):
    """Compute SSIM between two images."""
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    win_size = min(win_size, img1.shape[0], img1.shape[1])
    return ssim(img1, img2, win_size=win_size, channel_axis=-1, data_range=255)

def compute_fid(real_features, generated_features):
    """Compute FID score between real and generated features."""
    # Ensure inputs are 2D arrays
    if len(real_features.shape) == 1:
        real_features = real_features.reshape(1, -1)
    if len(generated_features.shape) == 1:
        generated_features = generated_features.reshape(1, -1)

    # Compute statistics
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False) if real_features.shape[0] > 1 else np.zeros((mu1.size, mu1.size))
    mu2 = np.mean(generated_features, axis=0)
    sigma2 = np.cov(generated_features, rowvar=False) if generated_features.shape[0] > 1 else np.zeros((mu2.size, mu2.size))

    # Calculate the mean squared difference
    ssdiff = np.sum((mu1 - mu2) ** 2)

    # Compute the square root of the product of the covariance matrices
    covmean = sqrtm(sigma1.dot(sigma2))

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def compute_kid(generated_features, real_features, degree=3, coef0=1, gamma=None):
    """Compute the Kernel Inception Distance (KID) score between generated and real features."""
    # Ensure 2D shape for features
    if len(real_features.shape) == 1:
        real_features = real_features.reshape(1, -1)
    if len(generated_features.shape) == 1:
        generated_features = generated_features.reshape(1, -1)

    # Compute polynomial kernel matrices
    K_real = polynomial_kernel(real_features, real_features, degree=degree, coef0=coef0, gamma=gamma)
    K_gen = polynomial_kernel(generated_features, generated_features, degree=degree, coef0=coef0, gamma=gamma)
    K_real_gen = polynomial_kernel(real_features, generated_features, degree=degree, coef0=coef0, gamma=gamma)

    m = K_real.shape[0]
    kid = (K_real.sum() + K_gen.sum() - 2 * K_real_gen.sum()) / (m * m)
    return np.abs(kid)

def get_sift_matches(img1, img2):
    """Get SIFT matches between two images."""
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY), None)
    keypoints2, descriptors2 = sift.detectAndCompute(cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY), None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    if descriptors1 is None or descriptors2 is None:
        return [], [], []
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    return points1, points2, matches

def get_essential_matrix(pose1, pose2, intrinsics):
    """Compute the essential matrix between two poses."""
    pose1_inv = np.linalg.inv(pose1)
    rel_pose = np.dot(pose1_inv, pose2)

    T = rel_pose[:3, 3]
    R = rel_pose[:3, :3]

    T_cross = np.array([
        [0, -T[2], T[1]],
        [T[2], 0, -T[0]],
        [-T[1], T[0], 0]
    ])

    E = np.dot(R, T_cross)
    E = np.dot(intrinsics.T, np.dot(E, intrinsics))

    return E

def get_min_dist(p1, E, kp):
    """Calculate the distance from a point to the epipolar line."""
    p1_h = np.append(p1, 1)

    # Calculate the epipolar line l' = E^T * p1
    epipolar_line = np.dot(E.T, p1_h)

    # Normalize the line equation
    a, b, c = epipolar_line
    norm = np.sqrt(a ** 2 + b ** 2)
    epipolar_line /= norm

    # Convert kp to homogeneous coordinates
    kp_h = np.append(kp, 1)

    # Calculate the distance from point to line
    distance = np.abs(np.dot(epipolar_line, kp_h)) / np.sqrt(epipolar_line[0] ** 2 + epipolar_line[1] ** 2)
    return distance

def compute_tsed(img1, img2, pose1, pose2, src_intrinsics, tar_intrinsics, threshold=12.0):
    """Compute the Temporal Self-Consistency Error Distance (TSED)."""
    points1, points2, matches = get_sift_matches(img1, img2)

    if len(matches) == 0:
        return 0, 1e8

    E12 = get_essential_matrix(pose1, pose2, src_intrinsics)
    E21 = get_essential_matrix(pose2, pose1, tar_intrinsics)

    seds = []
    for p1, p2 in zip(points1, points2):
        sed1 = get_min_dist(p1, E12, p2)
        sed2 = get_min_dist(p2, E21, p1)
        sed = 0.5 * (sed1 + sed2)
        seds.append(sed)

    seds_array = np.array(seds)
    below_threshold = seds_array < threshold
    count = np.sum(below_threshold)

    n_matches = len(seds)
    median_sed = np.median(seds_array) if n_matches > 0 else 1e8

    return count, median_sed

def tsed_evaluate(generated_frames, poses, intrinsics):
    """Evaluate TSED over a sequence of frames."""
    tsed_scores = []

    for i in range(len(generated_frames) - 1):
        gen_image1 = generated_frames[i]
        gen_image2 = generated_frames[i + 1]

        pose1 = poses[i]
        pose2 = poses[i + 1]
        intrinsic1 = intrinsics[i]
        intrinsic2 = intrinsics[i + 1]

        count, median_dist = compute_tsed(gen_image1, gen_image2, pose1, pose2, intrinsic1, intrinsic2)
        tsed_scores.append((count, median_dist))

    avg_tsed_count = np.mean([score[0] for score in tsed_scores])
    avg_tsed_dist = np.mean([score[1] for score in tsed_scores])

    med_tsed_count = np.median([score[0] for score in tsed_scores])
    med_tsed_dist = np.median([score[1] for score in tsed_scores])
    print(f'Median TSED distance: {med_tsed_dist}')
    return med_tsed_dist
    print(f'Average TSED count: {avg_tsed_count}')
    print(f'Average TSED distance: {avg_tsed_dist}')
    print(f'Median TSED count: {med_tsed_count}')
    print(f'Median TSED distance: {med_tsed_dist}')

##############################
# FVD Implementation Begins Here
##############################

class SequentialFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet50 and remove the final classification layer
        resnet = models.resnet50(pretrained=True)
        self.spatial_encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def encode_single_image(self, x):
        """Encode a single image to feature vector"""
        with torch.no_grad():
            features = self.spatial_encoder(x)
            # Flatten the features to 1D vector
            features = features.view(features.size(0), -1)
        return features

    def encode_sequence(self, sequence_features):
        """Encode a sequence of features temporally"""
        with torch.no_grad():
            sequence_features = sequence_features.view(-1, sequence_features.size(-1))
            temporal_features = self.temporal_encoder(sequence_features)
        return temporal_features

def process_single_image(image, transform, model, device):
    """Process a single image and return its features"""
    img_tensor = transform(image).unsqueeze(0).to(device)
    features = model.encode_single_image(img_tensor)
    return features.cpu().squeeze(0)  # Remove batch dimension

def extract_sequence_features(features_list, model, device, sequence_length=16):
    """Extract temporal features from a sequence of spatial features"""
    # Stack all features into a single tensor
    features_tensor = torch.stack(features_list)  # Shape: (num_frames, feature_dim)

    # Calculate number of complete sequences
    num_frames = features_tensor.size(0)
    num_sequences = (num_frames + sequence_length - 1) // sequence_length  # Ceiling division

    # Pad if necessary
    if num_frames < num_sequences * sequence_length:
        pad_size = num_sequences * sequence_length - num_frames
        padding = torch.zeros((pad_size, features_tensor.size(1)),
                              dtype=features_tensor.dtype,
                              device=features_tensor.device)
        features_tensor = torch.cat([features_tensor, padding], dim=0)

    # Reshape into sequences
    features_tensor = features_tensor.view(num_sequences, sequence_length, -1).to(device)

    # Process each sequence
    temporal_features = model.encode_sequence(features_tensor)
    return temporal_features

def compute_fvd(real_features, generated_features):
    """Compute Fréchet Video Distance between real and generated features."""
    # Ensure the features are 2D (num_sequences, feature_dim)
    if len(real_features.shape) > 2:
        real_features = real_features.reshape(real_features.shape[0], -1)
    if len(generated_features.shape) > 2:
        generated_features = generated_features.reshape(generated_features.shape[0], -1)

    real_features = real_features.cpu().numpy()
    generated_features = generated_features.cpu().numpy()

    # Calculate statistics
    real_mean = np.mean(real_features, axis=0)
    gen_mean = np.mean(generated_features, axis=0)

    real_cov = np.cov(real_features, rowvar=False) if real_features.shape[0] > 1 else np.zeros((real_features.shape[1], real_features.shape[1]))
    gen_cov = np.cov(generated_features, rowvar=False) if generated_features.shape[0] > 1 else np.zeros((generated_features.shape[1], generated_features.shape[1]))

    # Calculate FVD
    covmean, _ = sqrtm(real_cov.dot(gen_cov), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = np.sum((real_mean - gen_mean) ** 2) + np.trace(real_cov + gen_cov - 2 * covmean)
    return float(fvd)

def calculate_fvd_score(generated_frames, gt_frames, sequence_length=16, device='cpu'):
    """Calculate FVD score between generated and ground truth videos."""
    # Set up model and transforms
    model = SequentialFeatureExtractor().to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize using ImageNet mean and std
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Ensure equal number of frames
    min_frames = min(len(generated_frames), len(gt_frames))
    generated_frames = generated_frames[:min_frames]
    gt_frames = gt_frames[:min_frames]

    print(f"Processing {min_frames} frames for FVD calculation...")

    # Process generated frames
    print("Processing generated frames...")
    generated_features_list = []
    for image in tqdm(generated_frames):
        features = process_single_image(image, transform, model, device)
        generated_features_list.append(features)

    # Process ground truth frames
    print("Processing ground truth frames...")
    gt_features_list = []
    for image in tqdm(gt_frames):
        features = process_single_image(image, transform, model, device)
        gt_features_list.append(features)

    print("Extracting temporal features...")
    generated_temporal_features = extract_sequence_features(generated_features_list, model, device, sequence_length)
    gt_temporal_features = extract_sequence_features(gt_features_list, model, device, sequence_length)

    print("Computing FVD score...")
    fvd_score = compute_fvd(gt_temporal_features, generated_temporal_features)
    return fvd_score
