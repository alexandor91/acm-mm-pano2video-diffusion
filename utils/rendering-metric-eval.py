import os
import numpy as np
import torch
import lpips
import cv2
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tqdm import tqdm
from PIL import Image

# Load the LPIPS model
lpips_model = lpips.LPIPS(net='alex')

def center_crop_img_and_resize(src_image, image_size):
    while min(src_image.shape[:2]) >= 2 * image_size:
        new_size = (src_image.shape[1] // 2, src_image.shape[0] // 2)
        src_image = cv2.resize(src_image, new_size, interpolation=cv2.INTER_AREA)

    scale = image_size / min(src_image.shape[:2])
    new_size = (round(src_image.shape[1] * scale), round(src_image.shape[0] * scale))
    src_image = cv2.resize(src_image, new_size, interpolation=cv2.INTER_CUBIC)

    crop_y = (src_image.shape[0] - image_size) // 2
    crop_x = (src_image.shape[1] - image_size) // 2
    cropped_image = src_image[crop_y:crop_y + image_size, crop_x:crop_x + image_size]

    return cropped_image

def load_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Failed to load image: {file_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    return image

def load_depth_image(file_path):
    """Load depth image and convert to numpy array with float32 values"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Depth image file not found: {file_path}")
        
    if file_path.lower().endswith('.exr'):
        try:
            import OpenEXR
            import Imath
            exr_file = OpenEXR.InputFile(file_path)
            dw = exr_file.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            depth_str = exr_file.channel('R', pt)
            depth = np.frombuffer(depth_str, dtype=np.float32)
            depth.shape = (size[1], size[0])
        except ImportError:
            print("OpenEXR not installed. Falling back to cv2 for EXR files.")
            depth = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        depth = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    
    if depth is None:
        raise ValueError(f"Could not load depth image at {file_path}")
    
    depth = depth.astype(np.float32)
    
    # Handle potential invalid values
    depth[~np.isfinite(depth)] = 0
    
    return depth

def compute_lpips(img1, img2):
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    return lpips_model(img1, img2).item()

def compute_psnr(img1, img2):
    img1 = img1.permute(1, 2, 0).numpy()
    img2 = img2.permute(1, 2, 0).numpy()
    return psnr(img1, img2, data_range=img1.max() - img1.min())

def compute_ssim(img1, img2, win_size=3):
    img1 = img1.permute(1, 2, 0).numpy()
    img2 = img2.permute(1, 2, 0).numpy()
    win_size = min(win_size, img1.shape[0], img1.shape[1])
    return ssim(img1, img2, win_size=win_size, channel_axis=-1, data_range=img1.max() - img1.min())

def compute_depth_metrics(pred_depth, gt_depth, mask=None):
    """
    Compute RMSE and Absolute-relative error for depth images
    """
    if mask is None:
        mask = np.logical_and(
            np.logical_and(gt_depth > 0, pred_depth > 0),
            np.logical_and(np.isfinite(gt_depth), np.isfinite(pred_depth))
        )
    
    if not mask.any():
        return np.nan, np.nan
    
    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]
    
    rmse = np.sqrt(np.mean((pred_depth - gt_depth) ** 2))
    abs_rel = np.mean(np.abs(pred_depth - gt_depth) / gt_depth)
    
    return rmse, abs_rel

def evaluate(render_dir, ground_truth_dir):
    """Evaluate RGB images"""
    if not os.path.exists(render_dir) or not os.path.exists(ground_truth_dir):
        raise FileNotFoundError("One or both directories do not exist")

    render_files = sorted([f for f in os.listdir(render_dir) if f.endswith(('.png', '.jpg'))],
                         key=lambda x: int(os.path.splitext(x)[0]))
    ground_truth_files = sorted([f for f in os.listdir(ground_truth_dir) if f.endswith(('.png', '.jpg'))],
                               key=lambda x: int(os.path.splitext(x)[0]))

    if len(render_files) != len(ground_truth_files):
        raise ValueError(f"Number of files doesn't match: {len(render_files)} vs {len(ground_truth_files)}")

    lpips_scores = []
    psnr_scores = []
    ssim_scores = []

    for gen_file, gt_file in tqdm(zip(render_files, ground_truth_files), total=len(render_files)):
        gen_image = load_image(os.path.join(render_dir, gen_file))
        gt_image = load_image(os.path.join(ground_truth_dir, gt_file))
        
        lpips_score = compute_lpips(gen_image, gt_image)
        psnr_score = compute_psnr(gen_image, gt_image)
        ssim_score = compute_ssim(gen_image, gt_image)
        
        lpips_scores.append(lpips_score)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)

    return np.mean(lpips_scores), np.mean(psnr_scores), np.mean(ssim_scores)

def evaluate_depth(pred_dir, gt_dir):
    """Evaluate depth images"""
    if not os.path.exists(pred_dir) or not os.path.exists(gt_dir):
        raise FileNotFoundError("One or both depth directories do not exist")

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.exr'))],
                       key=lambda x: int(os.path.splitext(x)[0]))
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.exr'))],
                     key=lambda x: int(os.path.splitext(x)[0]))

    if len(pred_files) != len(gt_files):
        raise ValueError(f"Number of depth files doesn't match: {len(pred_files)} vs {len(gt_files)}")

    rmse_scores = []
    abs_rel_scores = []

    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files)):
        try:
            pred_depth = load_depth_image(os.path.join(pred_dir, pred_file))
            gt_depth = load_depth_image(os.path.join(gt_dir, gt_file))
            
            if pred_depth.shape != gt_depth.shape:
                pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            
            rmse, abs_rel = compute_depth_metrics(pred_depth, gt_depth)
            
            if not (np.isnan(rmse) or np.isnan(abs_rel)):
                rmse_scores.append(rmse)
                abs_rel_scores.append(abs_rel)
                
        except Exception as e:
            print(f"Error processing depth files {pred_file} and {gt_file}: {str(e)}")
            continue

    if not rmse_scores:
        raise ValueError("No valid depth comparisons were made")

    mean_rmse = np.mean(rmse_scores)
    mean_abs_rel = np.mean(abs_rel_scores)

    return mean_rmse, mean_abs_rel

if __name__ == "__main__":
    try:
        base_dir = '/home/student.unimelb.edu.au/xueyangk'
        
        # RGB evaluation paths
        pred_rgb_folder = 'fast-DiT/data/data/rgb'
        gt_rgb_folder = 'fast-DiT/data/data/rgb'
        
        # Depth evaluation paths
        pred_depth_folder = 'fast-DiT/data/data/depth'
        gt_depth_folder = 'fast-DiT/data/data/depth'

        render_dir = os.path.join(base_dir, pred_rgb_folder)
        ground_truth_dir = os.path.join(base_dir, gt_rgb_folder)
        depth_render_dir = os.path.join(base_dir, pred_depth_folder)
        depth_ground_truth_dir = os.path.join(base_dir, gt_depth_folder)

        print("Evaluating RGB images...")
        lpips_scores, psnr_scores, ssim_scores = evaluate(render_dir, ground_truth_dir)
        print(f'LPIPS: {lpips_scores:.4f}')
        print(f'PSNR: {psnr_scores:.4f}')
        print(f'SSIM: {ssim_scores:.4f}')

        print("\nEvaluating Depth images...")
        rmse_scores, abs_rel_scores = evaluate_depth(depth_render_dir, depth_ground_truth_dir)
        print(f'Mean RMSE: {rmse_scores:.4f}')
        print(f'Mean Absolute-Relative Error: {abs_rel_scores:.4f}')

    except Exception as e:
        print(f"An error occurred: {str(e)}")