import cv2
import numpy as np
import os

# Function to map perspective view to panorama
def map_perspective_to_panorama(perspective_img, FOV, THETA, PHI, panorama_size):
    # Initialize a black panorama image
    panorama = np.zeros((panorama_size[1], panorama_size[0], 3), dtype=np.uint8)
    
    # Get perspective image size
    height, width = perspective_img.shape[:2]
    
    # Intrinsic matrix (camera matrix) for the perspective view
    f = 0.5 * width / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], np.float32)
    K_inv = np.linalg.inv(K)
    
    # Create a meshgrid of pixel coordinates in the perspective image
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    
    # Combine to create 3D coordinates
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    xyz = xyz @ K_inv.T  # Transform into normalized 3D coordinates

    # Apply the rotation based on THETA and PHI to adjust for the camera's orientation
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)  # Rotation around Y-axis (THETA)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)  # Rotation around X-axis (PHI)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    xyz = xyz @ R.T

    # Convert 3D points to spherical coordinates (longitude, latitude)
    lon = np.arctan2(xyz[:, 0], xyz[:, 2])  # Longitude
    lat = np.arcsin(xyz[:, 1] / np.linalg.norm(xyz, axis=1))  # Latitude

    # Map spherical coordinates to panorama image coordinates
    lonlat = np.stack([lon, lat], axis=-1)
    lonlat[:, 0] = (lonlat[:, 0] / (2 * np.pi) + 0.5) * panorama_size[0]  # Longitude to X
    lonlat[:, 1] = (lonlat[:, 1] / np.pi + 0.5) * panorama_size[1]  # Latitude to Y
    
    # Remap perspective image to panorama
    lonlat = lonlat.reshape(height, width, 2).astype(np.float32)
    panorama = cv2.remap(perspective_img, lonlat[:, :, 0], lonlat[:, :, 1], 
                         interpolation=cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_CONSTANT, 
                         borderValue=(0, 0, 0))
    
    return panorama

basedir = 'C:\Users\anonymous\Downloads\PanaromaSamples2Video\Perspect&Pan'
filename = 'perspectimage.jpg'
path_dir = os.path.join(basedir, filename)
# Example Usage
perspective_img = cv2.imread(path_dir)
panorama_size = (2048, 1024)  # Set panorama size (width, height)

# Map the perspective image (FOV=90, THETA=0, PHI=0) to the front part of the panorama
panorama_img = map_perspective_to_panorama(perspective_img, FOV=90, THETA=0, PHI=0, panorama_size=panorama_size)

# Save or display the resulting panorama image
cv2.imwrite('mapped_panorama.jpg', panorama_img)
# cv2.imshow("Mapped Panorama", panorama_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()