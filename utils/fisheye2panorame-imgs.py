import os
import sys
import cv2
import numpy as np
import glob

class Perspective:
    def __init__(self, img_name , FOV, THETA, PHI ):

        tmp_img = cv2.imread(img_name, cv2.IMREAD_COLOR)

        # Check if the images are loaded 
        # if tmp_img is None:
        #     print("Error: Unable to load one or both images.")
        # else:
        #     # Resize both images to the specified size (1920x960)
        #     self._img = cv2.resize(tmp_img, (width, height))
        self._img = tmp_img
        [self._height, self._width, _] = self._img.shape
        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI
        self.hFOV = float(self._height) / self._width * FOV

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))

    

    def GetEquirec(self,height,width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        x,y = np.meshgrid(np.linspace(-180, 180,width),np.linspace(90,-90,height))
        
        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map,y_map,z_map),axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI))

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height , width, 3])
        inverse_mask = np.where(xyz[:,:,0]>0,1,0)

        xyz[:,:] = xyz[:,:]/np.repeat(xyz[:,:,0][:, :, np.newaxis], 3, axis=2)
        
        
        lon_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(xyz[:,:,1]+self.w_len)/2/self.w_len*self._width,0)
        lat_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(-xyz[:,:,2]+self.h_len)/2/self.h_len*self._height,0)
        mask = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),1,0)

        persp = cv2.remap(self._img, lon_map.astype(np.float32), lat_map.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        
        mask = mask * inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        persp = persp * mask
        
        
        return persp , mask
    
class MultiPerspective:
    def __init__(self, img_array , F_T_P_array ):
        
        assert len(img_array)==len(F_T_P_array)
        
        self.img_array = img_array
        self.F_T_P_array = F_T_P_array
    

    def GetEquirec(self,height,width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        merge_image = np.zeros((height,width,3))
        merge_mask = np.zeros((height,width,3))

        for img_dir,[F,T,P] in zip (self.img_array,self.F_T_P_array):
            per = Perspective(img_dir,F,T,P)        # Load equirectangular image
            img , mask = per.GetEquirec(height,width)   # Specify parameters(FOV, theta, phi, height, width)
            merge_image += img
            merge_mask +=mask

        merge_mask = np.where(merge_mask==0,1,merge_mask)
        merge_image = (np.divide(merge_image,merge_mask))

        
        return merge_image

class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        print(self._img.shape)
    

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        w_len = np.tan(np.radians(wFOV / 2.0))
        h_len = np.tan(np.radians(hFOV / 2.0))


        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len,width), [height,1])
        z_map = -np.tile(np.linspace(-h_len, h_len,height), [width,1]).T

        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.stack((x_map,y_map,z_map),axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)
        
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2])
        lon = np.arctan2(xyz[:, 1] , xyz[:, 0])

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180

        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90  * equ_cy + equ_cy

        
            
        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return persp
        
def panorama2cube(input_dir,output_dir):

    cube_size = 640

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    all_image = sorted(glob.glob(input_dir + '/*.*'))

    print(all_image)


    for index in range(len(all_image)):
        # image = '../Opensfm/source/library/test-1/frame{:d}.png'.format(i)
        equ = Equirectangular(all_image[index])    # Load equirectangular image
        #
        # FOV unit is degree
        # theta is z-axis angle(right direction is positive, left direction is negative)
        # phi is y-axis angle(up direction positive, down direction negative)
        # height and width is output image dimension
        #

        out_dir = output_dir + '/%02d/'%(index)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        img = equ.GetPerspective(90, 0, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output1 = out_dir +  'front.png'
        cv2.imwrite(output1, img)

        img = equ.GetPerspective(90, 90, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output2 = out_dir + 'right.png' 
        cv2.imwrite(output2, img)


        img = equ.GetPerspective(90, 180, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output3 = out_dir + 'back.png' 
        cv2.imwrite(output3, img)

        img = equ.GetPerspective(90, 270, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output4 = out_dir + 'left.png' 
        cv2.imwrite(output4, img)

        img = equ.GetPerspective(90, 0, 90, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output5 = out_dir + 'top.png' 
        cv2.imwrite(output5, img)

        img = equ.GetPerspective(90, 0, -90, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        output6 = out_dir + 'bottom.png' 
        cv2.imwrite(output6, img)


def cube2panorama(input_dir,output_dir):

    width = 1920
    height = 960

    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    
    front = input_dir + '/front.png'     ###### 640 x 640 size
    right = input_dir + '/right.png' 
    left = input_dir + '/left.png'
    back = input_dir + '/back.png'
    top = input_dir + '/top.png'
    bottom = input_dir + '/bottom.png'

    # this can turn cube to panorama
    per = MultiPerspective([front,right,back,left,top,bottom],
                            [[90, 0, 0],[90, 90, 0],[90, 180, 0],
                            [90, 270, 0],[90, 0, 90],[90, 0, -90]])    
    ########### persp size 1920 x 960 ################
    
    img = per.GetEquirec(height,width)  
    cv2.imwrite(output_dir, img)

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

if __name__ == '__main__':


    #
    # FOV unit is degree
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension
    #
    
    input_dir = "/home/student./anonymous"
    folder_type = "fast-DiT"
    output_dir = 'example'

    width = 1920
    height = 960

    persp_width = 640
    persp_height = 640

    basedir = "/home/student./anonymous"
    filename = "0a5c013435/dslr/resized_images"
    intrinsic_folder = "0a5c013435/dslr/colmap"

    # # Fisheye camera intrinsics (as provided)
    # K = np.array([[790.0312485027164, 0, 880.9192479122723],
    #             [0, 790.041861278764, 586.1276306268625],
    #             [0, 0, 1]])

    # D = np.array([-0.028024178435053528, -0.007290023031943852, 
    #             -0.00031718708743573923, -0.0004350185103994526])
    # Parse the camera intrinsic parameters from the text file
    intrinsic_params_dir = os.path.join(basedir, intrinsic_folder, 'cameras.txt')
    K, D, width, height = parse_camera_file(intrinsic_params_dir)
    
    if K is None or D is None:
        print("Error: Camera parameters not found.")

    # Get list of image files sorted by timestamp (filename)
    image_files = sorted(glob.glob(os.path.join(basedir, filename, "*.JPG")))
    # print(image_files)
    # Loop through each image
    for image_file in image_files:
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        # Load the fisheye image
        fisheye_image = cv2.imread(os.path.join(basedir, filename, image_file))
       
        # Get dimensions of the fisheye image
        height, width = fisheye_image.shape[:2]

        # Create an output image size for the undistorted image
        # Typically, this would be the same size as the input, but you can adjust if needed
        undistorted_size = (width, height)

        # Generate new camera matrix (if needed, can modify the balance parameter for scaling)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, undistorted_size, np.eye(3), balance=1)

        # Initialize maps for remapping the fisheye image to an undistorted perspective image
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, undistorted_size, cv2.CV_32FC1)

        # Undistort the fisheye image
        undistorted_image = cv2.remap(fisheye_image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        undist_fisheye_directory = os.path.join(input_dir, folder_type, output_dir, 'undis_fisheye.png')

        cv2.imwrite(undist_fisheye_directory, undistorted_image)

        # Define input directory and image size
        persp_dir = os.path.join(input_dir, folder_type, output_dir)  # Specify the correct input path

        # Define output filenames
        front = os.path.join(persp_dir, 'front.png')     # 640 x 640 size
        right = os.path.join(persp_dir, 'right.png')
        left = os.path.join(persp_dir, 'left.png')
        back = os.path.join(persp_dir, 'back.png')
        top = os.path.join(persp_dir, 'top.png')
        bottom = os.path.join(persp_dir, 'bottom.png')

        # Load the fisheye image
        # input1 = os.path.join(input_dir, folder_type, 'undis_fisheye.png')
        fisheye_image = undistorted_image ###cv2.imread(input_directory)

        # Check if the image is loaded correctly
        if fisheye_image is None:
            print("Error: Unable to load the image.")
        else:
            # Get the width and height of the loaded image
            height, width = fisheye_image.shape[:2]

            # Calculate the center width and the 1/4 width regions for the front image
            margin_x = width // 3
            half_margin_width = margin_x // 2
            center_x = width // 2
            # Split the image into left, right, and front
            left_image = fisheye_image[:, :margin_x]  # Left half
            right_image = fisheye_image[:, 2*margin_x:]  # Right half
            front_image = fisheye_image[:, center_x - half_margin_width: center_x + half_margin_width]  # Center region (1/4 width)

            # Resize each region to 640x640
            resized_left = cv2.resize(left_image, (640, 640))
            resized_right = cv2.resize(right_image, (640, 640))
            resized_front = cv2.resize(front_image, (640, 640))

            # Save the resized images
            cv2.imwrite(left, resized_left)
            cv2.imwrite(right, resized_right)
            cv2.imwrite(front, resized_front)

            # Create black images for back, top, and bottom
            black_image = np.zeros((640, 640, 3), dtype=np.uint8)

            # Save the black images
            cv2.imwrite(back, black_image)
            cv2.imwrite(top, black_image)
            cv2.imwrite(bottom, black_image)

            print("Images have been split, resized, and saved successfully.")

            cube2panorama(persp_dir, os.path.join(persp_dir, 'panoutput_' + base_name + '.png'))
   # input_dir2 = os.path.join(input_dir, folder_type, 'undis_fisheye.png')

    # # Load the image from the input path
    # image1 = cv2.imread(input_dir1)
    # image2 = cv2.imread(input_dir2)

    # # Check if the images or image2 is None:
    #     print("Error: are loaded correctly
    # if image1 is None Unable to load one or both images.")
    # else:
    #     # Resize both images to the specified size (1920x960)
    #     input1 = cv2.resize(image1, (width, height))
    #     input2 = cv2.resize(image2, (width, height))

    # this can turn cube to panorama
    # equ = MultiPerspective([input_dir1,input_dir2],
    #                         [[120, 0, 0],[120, 0, 90]])    
    
    
    # img = equ.GetEquirec(height,width)  
    # print(img.shape)


# class Perspective:
    # def __init__(self, img_name , FOV, THETA, PHI ):
    #     self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    #     [self._height, self._width, _] = self._img.shape
    #     self.FOV = FOV
    #     self.THETA = THETA
    #     self.PHI = PHI
    

    # def GetEquirec(self,height,width):
    #     #
    #     # THETA is left/right angle, PHI is up/down angle, both in degree
    #     #

    #     equ_h = height
    #     equ_w = width
    #     equ_cx = (equ_w - 1) / 2.0
    #     equ_cy = (equ_h - 1) / 2.0

    #     wFOV = self.FOV
    #     hFOV = float(self._height) / self._width * wFOV

    #     w_len = np.tan(np.radians(wFOV / 2.0))
    #     h_len = np.tan(np.radians(hFOV / 2.0))


    #     x_map = np.ones([self._height, self._width], np.float32)
    #     y_map = np.tile(np.linspace(-w_len, w_len,self._width), [self._height,1])
    #     z_map = -np.tile(np.linspace(-h_len, h_len,self._height), [self._width,1]).T

    #     print(z_map[0])

    #     D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    #     xyz = np.stack((x_map,y_map,z_map),axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)
    #     print(xyz[0,:,2])
        
    #     y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    #     z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    #     [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.THETA))
    #     [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI))

    #     xyz = xyz.reshape([self._height * self._width, 3]).T
    #     xyz = np.dot(R1, xyz)
    #     xyz = np.dot(R2, xyz).T
    #     lat = np.arcsin(xyz[:, 2])
    #     lon = np.arctan2(xyz[:, 1] , xyz[:, 0])

    #     lon = lon / np.pi * 180
    #     lat = -lat / np.pi * 180

    #     print(lat.reshape([self._height , self._width])[0])
    #     print(lon.reshape([self._height , self._width])[0])
        
    #     lon = (lon / 180 * equ_cx + equ_cx).astype(np.int)
    #     lat = (lat / 90  * equ_cy + equ_cy).astype(np.int)
    #     coordinate = (lat,lon)

    #     x_map = np.repeat(np.arange(self._height), self._width)
    #     y_map = np.tile(np.arange(self._width), self._height)

    #     blank_map_x = np.zeros((height,width))
    #     blank_map_y = np.zeros((height,width))
    #     mask = np.zeros((height,width,3))

    #     blank_map_x[coordinate] = x_map
    #     blank_map_y[coordinate] = y_map
    #     mask[coordinate] = [1,1,1]

    #     # print(lat.reshape([self._height, self._width]))
    #     # print(lon.reshape([self._height, self._width])[-1,1910:1930])


    #     persp = cv2.remap(self._img, blank_map_y.astype(np.float32), blank_map_x.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        
    #     persp = persp * mask
        
    #     return persp , mask
        