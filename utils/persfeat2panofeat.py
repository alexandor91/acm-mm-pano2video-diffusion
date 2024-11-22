import numpy as np
import os
import einops as E
from PIL import Image 
from sklearn.manifold import TSNE
from scipy.ndimage import zoom
import cv2
import matplotlib.pyplot as plt
import json
import glob

def project_and_save_tsne_image(src_features, output_path='tsne_src_viz.png'):
    """
    Projects the valid features (derived mask) from 4 dimensions to 3 dimensions using t-SNE and saves it as an image.

    Args:
    - src_features (numpy array): The input array with shape (1, 4, 32, 40).
    - output_path (str): The path where the output image will be saved.
    """
    # Remove the batch dimension
    if src_features.ndim == 4:
        src_features = src_features[0]  # Shape: (4, 32, 40)

    # Get height and width of the feature map
    H, W = src_features.shape[1], src_features.shape[2]

    # Derive the binary mask (1 where features are non-zero, 0 otherwise)
    mask = np.any(src_features != 0, axis=0).astype(np.uint8)  # Shape: (32, 40)

    # Reshape src_features to (1280, 4) and mask to (1280,)
    src_features_reshaped = src_features.reshape(4, -1).T  # Shape: (1280, 4)
    mask_flat = mask.flatten()  # Shape: (1280,)

    # Extract only the valid features (where mask == 1)
    valid_features = src_features_reshaped[mask_flat == 1]  # Shape: (num_valid_pixels, 4)

    # Perform t-SNE on valid features to reduce from 4 channels to 3
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(valid_features)  # Shape: (num_valid_pixels, 3)

    # Create an empty result array for the full feature map
    tsne_image = np.zeros((H * W, 3))

    # Fill valid pixels with t-SNE results
    tsne_image[mask_flat == 1] = tsne_result

    # Reshape the result back to the original image format (32, 40, 3)
    tsne_image = tsne_image.reshape(H, W, 3)

    # Normalize to range [0, 255] for visualization
    tsne_image = (tsne_image - tsne_image.min()) / (tsne_image.max() - tsne_image.min()) * 255
    tsne_image = tsne_image.astype(np.uint8)

    # Save the image using OpenCV
    cv2.imwrite(output_path, tsne_image)
    return tsne_image

class PerspectiveFeatureMap:
    def __init__(self, feature_map, FOV, THETA, PHI):
        self._feature_map = feature_map
        self._channels, self._height, self._width = feature_map.shape
        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI
        self.hFOV = float(self._height) / self._width * FOV

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))

    def GetEquirec(self, height, width):
        # Create the panorama coordinate map
        x, y = np.meshgrid(np.linspace(-180, 180, width), np.linspace(90, -90, height))

        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map, y_map, z_map), axis=2)

        # Rotation matrices
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        R1, _ = cv2.Rodrigues(z_axis * np.radians(self.THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI))

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        # Transform coordinates
        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height, width, 3])
        inverse_mask = np.where(xyz[:, :, 0] > 0, 1, 0)

        # Normalize the xyz values
        xyz[:, :] = xyz[:, :] / np.repeat(xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)

        # Create longitude and latitude maps
        lon_map = np.where((-self.w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < self.w_len) &
                           (-self.h_len < xyz[:, :, 2]) & (xyz[:, :, 2] < self.h_len),
                           (xyz[:, :, 1] + self.w_len) / 2 / self.w_len * self._width, 0)
        lat_map = np.where((-self.w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < self.w_len) &
                           (-self.h_len < xyz[:, :, 2]) & (xyz[:, :, 2] < self.h_len),
                           (-xyz[:, :, 2] + self.h_len) / 2 / self.h_len * self._height, 0)
        mask = np.where((-self.w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < self.w_len) &
                        (-self.h_len < xyz[:, :, 2]) & (xyz[:, :, 2] < self.h_len), 1, 0)

        # Remap the feature maps for each channel
        persp = np.zeros((self._channels, height, width))
        for ch in range(self._channels):
            persp[ch] = cv2.remap(self._feature_map[ch], lon_map.astype(np.float32),
                                  lat_map.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        mask = mask * inverse_mask
        print("&&&&&&&& @@@@@@@@@@@@ &&&&&")
        print(persp.shape)
        return persp * mask, mask

class MultiPerspectiveFeatureMap:
    def __init__(self, feature_maps, F_T_P_array):
        assert len(feature_maps) == len(F_T_P_array)
        self.feature_maps = feature_maps
        self.F_T_P_array = F_T_P_array

    def GetEquirec(self, height, width):
        # if self.feature_maps.ndim == 4:
        #     self.feature_maps = self.feature_maps[0]  # Shape: (4, 32, 40)

        # Initialize the panorama feature map with zeros (channels x height x width)
        merge_feature_map = np.zeros((self.feature_maps[0].shape[0], height, width))  # 4x32x64 for panorama
        merge_mask = np.zeros((height, width))

        # Merge the feature maps into the panorama
        for feature_map, [F, T, P] in zip(self.feature_maps, self.F_T_P_array):
            per = PerspectiveFeatureMap(feature_map, F, T, P)
            img, mask = per.GetEquirec(height, width)
            print("##### multi img, mask #####")
            print(img.shape)
            print(mask.shape)
            merge_feature_map += img
            merge_mask += mask

        # Prevent division by zero in areas without data
        merge_mask = np.where(merge_mask == 0, 1, merge_mask)
        merge_feature_map = np.divide(merge_feature_map, merge_mask)
        print("##### merged img, mask #####")
        print(merge_feature_map.shape)
        print(merge_mask.shape)

        return merge_feature_map
    
def resize_feature_map_numpy(feature_map, new_shape):
    """Resize the feature map using NumPy operations."""
    # Calculate the scaling factor
    scale = new_shape[2] / feature_map.shape[2]
    
    # Create indices for the new array
    orig_indices = np.arange(feature_map.shape[2])
    new_indices = np.arange(0, feature_map.shape[2], 1/scale)
    
    # Use linear interpolation
    resized = np.zeros(new_shape)
    for i in range(feature_map.shape[0]):
        for j in range(feature_map.shape[1]):
            resized[i, j] = np.interp(new_indices, orig_indices, feature_map[i, j])
    
    return resized

def resize_and_pad_feature_map(feature_map, intermediate_shape, final_shape):
    """Resize the feature map and pad it to the original size."""
    # Resize the feature map
    resized_feature_map = resize_feature_map_numpy(feature_map, intermediate_shape)

    # Calculate padding
    pad_width = final_shape[2] - intermediate_shape[2]
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    # Pad the resized feature map
    padded_feature_map = np.pad(resized_feature_map, 
                                ((0, 0), (0, 0), (left_pad, right_pad)), 
                                mode='constant', constant_values=0)

    return padded_feature_map

def cube2panorama_feature_map(front_map, right_map, back_map, left_map, top_map, bottom_map):
    """
    Converts 6 cube feature maps into a panorama feature map (4x32x64).
    """
    height = 32  #### panorama feature size #########
    width = 64   ##### panorama feature size #########

    # Define the FOV, THETA, PHI for each direction
    cube_params = [[90, 0, 0],  # Front
                   [90, 90, 0],  # Right
                   [90, 180, 0],  # Back
                   [90, 270, 0],  # Left
                   [90, 0, 90],  # Top
                   [90, 0, -90]]  # Bottom

    # Load all feature maps into a list
    cube_maps = [front_map, right_map, back_map, left_map, top_map, bottom_map]

    # Perform cube to panorama conversion
    per = MultiPerspectiveFeatureMap(cube_maps, cube_params)
    panorama_feature_map = per.GetEquirec(height, width)

    return panorama_feature_map

# Parse the scannet++ cameras.txt file to extract camera parameters
def parse_camera_file(camera_file):
    with open(camera_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith('#'):
                parts = line.split()
                if len(parts) > 6:  # Ensure there are enough fields
                    # Extract the relevant camera parameters
                    camera_model = parts[1]
                    width = int(parts[2])
                    height = int(parts[3])
                    params = list(map(float, parts[4:]))
                    K = np.array([[params[0], 0, params[2]],
                                  [0, params[1], params[3]],
                                  [0, 0, 1]])
                    D = np.array(params[4:])
                    return K, D, width, height
    return None, None, None, None

if __name__ == "__main__":
    # Download all DiT checkpoints
    print("###### main starts#########")
    base_dir = '/home/student./anonymous'
    folder_type = 'PerspectAndPano'
    filename = '0.jpg'
    pan_vae_features = 'panofeature.npy'

    filename2 = '70.jpg'
    pers_vae_features = 'perspectfeature.npy'

    ###########load VAE numpy features################
    src_feats = np.load(os.path.join(base_dir, folder_type, pers_vae_features))
    tar_feats = np.load(os.path.join(base_dir, folder_type, pan_vae_features))


    # Remove the batch dimension
    if src_feats.ndim == 4:
        src_feats = src_feats[0]  # Shape: (4, 32, 40)
    # Remove the batch dimension
    if tar_feats.ndim == 4:
        tar_feats = tar_feats[0]  # Shape: (4, 32, 40)

    tsne_image1 = project_and_save_tsne_image(src_feats, output_path='tsne_pers_feature_viz.png')
    tsne_image2 = project_and_save_tsne_image(tar_feats, output_path='tsne_pan_feature_viz.png')

    # src_image_path = os.path.join(base_dir, folder_type, filename)
    # tgt_image_path = os.path.join(base_dir, folder_type, filename2)

    # Example usage:
    # input_feature_map = np.random.rand(4, 32, 40)  # Simulating a feature map with shape (4, 32, 40)
    FOV = 90
    THETA = 0
    PHI = 0


    print(src_feats.shape)
    print(tar_feats.shape)

    # Get the width and height of the loaded image
    # src_feats = src_feats.transpose(1, 2, 0)  # Moves the dimensions

    height, width = src_feats.shape[1:3]

    # Calculate the center width and the 1/4 width regions for the front image
    margin_x = width // 3
    half_margin_width = margin_x // 2
    center_x = width // 2


    
    right_feats = src_feats  # Right feature map
    # Example usage
    intermediate_shape = (4, height, width//2)
    final_shape = (4, height, width)

    right_feats_resized = resize_and_pad_feature_map(right_feats, intermediate_shape, final_shape)
    print(right_feats.shape)  # Output: (4, 32, 40)

    # Split the image into left, right, and front
    left_feats =  np.zeros((4, height, width), dtype=np.uint8)  # Left half
    front_feats = np.zeros((4, height, width), dtype=np.uint8) #np.zeros((4, height, width), dtype=np.uint8) # Center region (1/4 width)
    back_feats = np.zeros((4, height, width), dtype=np.uint8)  #np.zeros((4, height, width), dtype=np.uint8)   #np.zeros((height, width, 4), dtype=np.uint8)
    top_feats = np.zeros((4, height, width), dtype=np.uint8)    
    bottom_feats = np.zeros((4, height, width), dtype=np.uint8)

    print("Images have been split, resized, and saved successfully.")

    print(front_feats.shape)
    panorama_feature_map = cube2panorama_feature_map(front_feats, right_feats, back_feats, left_feats, top_feats, bottom_feats)

    # pano_height, pano_width = panorama_feature_map.shape[:2]
    # # Calculate the zoom factors for height and width (no zoom for the 4 channels)
    # zoom_factors = (32 / pano_height, 64 / pano_width, 4)  # Resizing only height and width, keeping channel dimension the same

    print(panorama_feature_map.shape)
    # Resize using zoom
    # resized_panorama_feature_map = zoom(panorama_feature_map, zoom_factors, order=1)  # order=1 for bilinear interpolation
    # resized_panorama_feature_map = resized_panorama_feature_map.transpose(2, 0, 1)  # Moves the dimensions

    # perspective_map = PerspectiveFeatureMap(src_feats, FOV, THETA, PHI)
    # output_feature_map = perspective_map.get_equirec_feature_map(32, 64)  # Output will be (4, 32, 64)

    project_and_save_tsne_image(panorama_feature_map, output_path='persp2pano.png')
    # src_image = load_resize_image_cv2(src_image_path)
    # tgt_image = load_resize_image_cv2(tgt_image_path)
