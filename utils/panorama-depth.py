import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import os

# Load Depth Anything v2 model
model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14_512')
model.eval()

# Configure transform
transform = [
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='minimal',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
]

def estimate_depth(image_path):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    # Apply transforms
    for t in transform:
        img = t(img)

    # Add batch dimension
    img = torch.from_numpy(img).unsqueeze(0)

    # Inference
    with torch.no_grad():
        depth = model(img)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=img.shape[2:],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = depth.cpu().numpy()
    
    # Normalize the depth map
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    
    return depth

def save_depth_map(depth_map, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(depth_map, cmap='plasma')
    plt.axis('off')
    plt.colorbar(label='Depth')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    output_depth_file = "depth_0e92a69a50414253a23043758f111cec.png"
    base_dir = "/home/student./anonymous/"
    folder = "Equirec2Perspec/PanaromaSamples"
    filename = "0e92a69a50414253a23043758f111cec.jpg"
    input_image_path = os.path.join(base_dir, folder, filename)    
    output_image_path = os.path.join(base_dir, folder, filename)    

    depth_map = estimate_depth(input_image_path)
    save_depth_map(depth_map, output_image_path)
    print(f"Depth map saved to {output_image_path}")