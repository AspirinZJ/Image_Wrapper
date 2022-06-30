'''
This is a python package using Scaramuzza and Mei's method to rectify omnidirectional images

Reference:
1.  https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab
2.  https://docs.opencv.org/4.x/dd/d12/tutorial_omnidir_calib_main.html 
'''


import cv2 as cv
import math
import numpy as np
import yaml


def preprocess_img(src):
    '''
    Crop the image. Only the circle area is used.

    Input:
        src (numpy.ndarray) : Input uncropped image 
    Output:
        dst (numpy.ndarray) : Cropped image with only omnidirectional FOV
    '''
    return src[80:, 310:1540]


class SCARA_OCAM_MODEL(object):
    '''
    Scaramuzza Ocam model

    The calibration is done in Matlab. 
    Calibration parameters is stored in calib_result.txt
    '''

    def __init__(self, fname=""):
        self.xc = 0         # optical center
        self.yc = 0         # optical center
        self.c = 0          # affine coefficient
        self.d = 0          # affine coefficient
        self.e = 0          # affine coefficient
        self.invpol = []    # inverse f function coefficients. Used in world2cam
        self.ss = []        # f function coefficients. Used in cam2world
        self.shape = []     # img shape "height" and "width

        if fname != "":
            self.read_result(fname)

    def read_result(self, fname):
        '''
        Read parameters from file

        Input:
            fname (str) : path of scara.yaml
        Output:
            None
        '''
        with open(fname, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)

        self.ss = np.array(data['ss'], dtype=np.float64)
        self.invpol = np.array(data['invpol'], dtype=np.float64)
        self.shape = np.array(data['shape'], dtype=np.float64)
        self.c, self.d, self.e = data['cde']
        self.xc, self.yc = data['xy']

    def create_panoramic_undistortion_LUT(self, Rmin, Rmax, new_img_size):
        '''
        Create Look Up Table (LUT) for remapping omni image into panoramic image 

        Input:
            Rmin (int) : Smallest radius to cut from optical center (xc, yc)
            Rmax (int) : Largest radius to cut from optical center (xc, yc)
            new_img_size (list) : size of new image [height, width]
        Output:
            mapx (numpy.ndarray) : mapx to be used in np.remap
            mapy (numpy.ndarray) : mapy to be used in np.remap
        '''
        height = new_img_size[0]
        width = new_img_size[1]

        mapx = np.zeros((height, width), dtype=np.float32)
        mapy = np.zeros((height, width), dtype=np.float32)

        for i in range(height):
            for j in range(width):
                # Note, if you would like to flip the image, just inverte the sign of theta
                theta = (j) / width*2 * math.pi
                rho = Rmax - (Rmax-Rmin) / height * i
                mapx[i][j] = self.yc + rho * math.sin(theta)
                mapy[i][j] = self.xc + rho * math.cos(theta)

        return mapx, mapy

    def panoramic_rectify(self, src, Rmax, Rmin, new_img_size):
        '''
        Rectify omnidirectional image into panoramic image 

        Input:
            src (numpy.ndarray) : Input omnidirectional image 
            Rmin (int) : Smallest radius to cut from optical center (xc, yc)
            Rmax (int) : Largest radius to cut from optical center (xc, yc)
            new_img_size (list) : size of new image [height, width]
        Output:
            img_rectified (numpy.ndarray) : Rectified panoramic image
        '''
        # Create panoramic look up table
        map_x, map_y = self.create_panoramic_undistortion_LUT(
            Rmax, Rmin, new_img_size)
        # Remap into panoramic image
        dst = cv.remap(src, map_x, map_y, cv.INTER_CUBIC)
        # Rotate 180 degree to align ground to the bottom
        img_rectified = cv.rotate(dst, cv.ROTATE_180)

        return img_rectified


class MEI_OCAM_MODEL(object):
    '''
    Mei Ocam model from opencv.
    '''

    def __init__(self, fname=""):
        self.K = np.zeros((3, 3))  # Intrinsic Matrix
        self.D = np.zeros((1, 4))  # Distortion coefficients [k1, k2, p1, p2]
        self.Xi = np.zeros((1, 1))  # Mei's Model coefficient

        if fname != '':
            self.read_result(fname)

    def read_result(self, fname):
        '''
        Read config from file

        Input:
            fname (str) : path of mei.yaml
        Output:
            None
        '''
        with open(fname, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)

        self.K = np.array(data['K'], dtype=np.float64).reshape((3, 3))
        self.D = np.array([data['D']], dtype=np.float64)
        self.Xi = np.array([data['Xi']], dtype=np.float64)

    def panoramic_rectify(self, src, new_img_size):
        '''
        Rectify omni image into panoramic image 

        Input:
            src (numpy.ndarray) : Input omnidirectional image 
            new_img_size (list) : size of new image [width, height]
        Output:
            dst (numpy.ndarray) : Rectified panoramic image
        '''
        # Rotated 180 to let the front fit in the middle
        r = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)

        # Create panoramic look up table
        map1, map2 = cv.omnidir.initUndistortRectifyMap(
            self.K,
            self.D,
            self.Xi,
            r, self.K, new_img_size, cv.CV_16SC2, cv.omnidir.RECTIFY_CYLINDRICAL)
        # Rotated 180 to let the front fit in the middle

        dst = cv.remap(src, map1, map2, cv.INTER_CUBIC)

        return dst


if __name__ == "__main__":
    # example usage
    # Note that new_image_size height and width is different due to different conventions 

    original_img = cv.imread("example.jpg")
    print(f'Origianl Image has size of {original_img.shape}')

    cropped = preprocess_img(original_img)
    print(f'Cropped Image has size of {cropped.shape}')

    cv.imwrite("cropped.jpg", cropped)

    scara = SCARA_OCAM_MODEL("scara.yaml")
    mei = MEI_OCAM_MODEL("mei.yaml")

    res_scara = scara.panoramic_rectify(cropped, 540, 200, (400, 1800))
    res_mei = mei.panoramic_rectify(cropped, (2900, 800))

    cv.imwrite("scara.jpg", res_scara)
    cv.imwrite("mei.jpg", res_mei)

