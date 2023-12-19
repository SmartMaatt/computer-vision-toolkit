import os
import sys
import glob
import argparse
import json
from json import JSONEncoder

import cv2 as cv
import numpy as np

# Defining the dimensions of chessboard
CHESSBOARD = (7,10)
SINGLE_SQUARE = 50
IMAGES_PATH_CAM1 = "..\..\\datasets\s4\cam2"
IMAGES_PATH_CAM2 = "..\..\\datasets\s4\cam3"
OUTPUT_PATH_CAM1 = "multicam_calibration_cam1.json"
OUTPUT_PATH_CAM2 = "multicam_calibration_cam2.json"


# Performs camera calibration procedure
def calibration_procedure():
    #TermCriteria (int type, int maxCount, double epsilon) 
    # Criteria for termination of the iterative process of corner refinement. 
    # That is, the process of corner position refinement stops either after criteria.maxCount iterations 
    # or when the corner position moves by less than criteria.epsilon on some iteration.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)                                 # Define [70][3] array filled with zeros
    objp[:,:2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * SINGLE_SQUARE        # Fills indexies to elements
    image_size = None

    # Arrays to store object points and image points from all the images
    objpoints_cam1 = [] # 3d point in real world space
    imgpoints_cam1 = [] # 2d points in image plane [Corner sub pix]
    objpoints_cam2 = [] # 3d point in real world space
    imgpoints_cam2 = [] # 2d points in image plane [Corner sub pix]

    counter = 0
    accepted_counter = 0
    print('Starting chessboard detection...')

    # Extracting paths of images stored in a given directory
    images_cam1 = glob.glob(IMAGES_PATH_CAM1 + "\*.png")
    images_cam2 = glob.glob(IMAGES_PATH_CAM2 + "\*.png")

    # Removing uncommon files
    images_cam1, images_cam2 = remove_uncommon_images(images_cam1, images_cam2)
    loop_counter = len(images_cam1)

    for i in range(loop_counter):
        img_cam1 = cv.imread(images_cam1[i])
        img_cam2 = cv.imread(images_cam2[i])

        gray_cam1 = cv.cvtColor(img_cam1, cv.COLOR_BGR2GRAY)
        gray_cam2 = cv.cvtColor(img_cam2, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret_cam1, corners_cam1 = cv.findChessboardCorners(gray_cam1, CHESSBOARD, None)
        ret_cam2, corners_cam2 = cv.findChessboardCorners(gray_cam2, CHESSBOARD, None)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of chessboard
        """
        counter += 1
        image_size_cam1 = gray_cam1.shape[::-1]
        image_size_cam2 = gray_cam2.shape[::-1]

        if image_size_cam1 != image_size_cam2:
            print('Couple rejected due to diffrent image sizes')
            continue

        # Only if both found
        if ret_cam1 == True and ret_cam2 == True:
            # Refining pixel coordinates for given 2d points.
            cornersSubPix_cam1 = cv.cornerSubPix(gray_cam1, corners_cam1, (11, 11), (-1, -1), criteria)
            cornersSubPix_cam2 = cv.cornerSubPix(gray_cam2, corners_cam2, (11, 11), (-1, -1), criteria)

            objpoints_cam1.append(objp)
            objpoints_cam2.append(objp)

            imgpoints_cam1.append(cornersSubPix_cam1)
            imgpoints_cam2.append(cornersSubPix_cam2)

            accepted_counter += 1
            print(f'Chessboard detection succeeded: [{counter}] {images_cam1[i]}')
            print(f'Chessboard detection succeeded: [{counter}] {images_cam2[i]}')
            print('Couple accepted')
        elif ret_cam1 == True and ret_cam2 == False:
            print(f'Chessboard detection succeeded: [{counter}] {images_cam1[i]}')
            print(f'Chessboard detection failed: [{counter}] {images_cam2[i]}')
            print('Couple rejected')
        elif ret_cam1 == False and ret_cam2 == True:
            print(f'Chessboard detection failed: [{counter}] {images_cam1[i]}')
            print(f'Chessboard detection succeeded: [{counter}] {images_cam2[i]}')
            print('Couple rejected')
        else:
            print(f'Chessboard detection failed: [{counter}] {images_cam1[i]}')
            print(f'Chessboard detection failed: [{counter}] {images_cam2[i]}')
            print('Couple rejected')


    if accepted_counter == 0:
        print("No images accepted, calibration aborted")
        sys.exit()

    print("\n>>> Calibrating camera one... <<<")
    mtx1, dist1, rvecs1, tvecs1, objpoints1, imgpoints1 = calibrate_camera(objpoints_cam1, imgpoints_cam1, image_size_cam1)

    print("\n>>> Calibrating camera two... <<<")
    mtx2, dist2, rvecs2, tvecs2, objpoints2, imgpoints2 = calibrate_camera(objpoints_cam2, imgpoints_cam2, image_size_cam2)

    return (mtx1, dist1, rvecs1, tvecs1, objpoints1, imgpoints1, image_size_cam1, mtx2, dist2, rvecs2, tvecs2, objpoints2, imgpoints2, image_size_cam2)


# Calibrates camera based on input parametes
def calibrate_camera(objpoints, imgpoints, image_size):
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    if ret:
        print("Camera calibration succeeded")
    else:
        print("Camera calibration failed")
        sys.exit()

    print("Camera matrix:")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)

    # Object points conversion
    for i, objpoint in enumerate(objpoints):
        objpoints[i] = np.asarray(objpoint, dtype=np.float32)

    # Image points conversion
    for i, imgpoint in enumerate(imgpoints):
        imgpoints[i] = np.asarray(imgpoint, dtype=np.float32)

    return (mtx, dist, rvecs, tvecs, objpoints, imgpoints)


# Remove uncommon photos from calibration
def remove_uncommon_images(images_cam1, images_cam2):
    folder_cam1 = os.path.split(images_cam1[0])[0]
    folder_cam2 = os.path.split(images_cam2[0])[0]

    split_img_cam1 = [os.path.basename(x) for x in images_cam1]
    split_img_cam2 = [os.path.basename(x) for x in images_cam2]

    common_file_names = [element for element in split_img_cam1 if element in split_img_cam2]
 
    new_img_cam1 = [os.path.join(folder_cam1, x) for x in common_file_names]
    new_img_cam2 = [os.path.join(folder_cam2, x) for x in common_file_names]

    print(f"\nRemoved {max(len(images_cam1), len(images_cam2)) - len(common_file_names)} uncommon files")
    return (new_img_cam1, new_img_cam2)


# Save final parameter matrix to json
def save_to_json(mtx, dist, rvecs, tvecs, image_size, objpoints, imgpoints, output_path):
    # >>> Saving result to json <<<
    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    json_result = {
        "mtx": mtx,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "imageSize": image_size,
        "objPoints": objpoints,
        "imgPoints": imgpoints
    }

    # Writing to json
    with open(output_path, "w") as outfile:
        json.dump(json_result, outfile, indent=4, cls=NumpyArrayEncoder)


# Parses and interprets console line arguments
def parse_argumetns():
    global CHESSBOARD
    global SINGLE_SQUARE
    global IMAGES_PATH_CAM1
    global IMAGES_PATH_CAM2
    global OUTPUT_PATH_CAM1
    global OUTPUT_PATH_CAM2

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--h', type=int, default=CHESSBOARD[0], metavar='Height', help='Height of checkerboard')
        parser.add_argument('--w', type=int, default=CHESSBOARD[1], metavar='Width', help='Width of checkerboard')
        parser.add_argument('--s', type=float, default=SINGLE_SQUARE, metavar='Square size', help='Real size of square')

        parser.add_argument('--i1', type=str, default=IMAGES_PATH_CAM1, metavar='Input cam1 path', help='Path to input cam1 image dataset')
        parser.add_argument('--i2', type=str, default=IMAGES_PATH_CAM2, metavar='Input cam2 path', help='Path to input cam2 image dataset')
        parser.add_argument('--o1', type=str, default=OUTPUT_PATH_CAM1, metavar='Output cam1 path', help='Path to output cam1 configuration json file')
        parser.add_argument('--o2', type=str, default=OUTPUT_PATH_CAM2, metavar='Output cam2 path', help='Path to output cam2 configuration json file')
    except argparse.ArgumentError as ex:
        print(ex)
        sys.exit()

    # Save arguments to global variables
    args = parser.parse_args()
    print('>>> Inserted arguments <<<')
    CHESSBOARD = (args.h, args.w)
    print(f'Chessboard size: ({CHESSBOARD[0]}, {CHESSBOARD[1]})')
    SINGLE_SQUARE = args.s
    print(f'Real square size: {SINGLE_SQUARE}mm')
    IMAGES_PATH_CAM1 = args.i1
    print(f'Input cam1 images folder path: {IMAGES_PATH_CAM1}')
    IMAGES_PATH_CAM2 = args.i2
    print(f'Input cam2 image folder path: {IMAGES_PATH_CAM2}')
    OUTPUT_PATH_CAM1 = args.o1
    print(f'Output cam1 data json path: {OUTPUT_PATH_CAM1}')
    OUTPUT_PATH_CAM2 = args.o2
    print(f'Output cam2 data json path: {OUTPUT_PATH_CAM2}')
    print('')


# Validates console line arguments
def validate_arguments():
    # Check integer arguments
    if CHESSBOARD[0] <= 0 or CHESSBOARD[1] <= 0:
        print('Checkerboard size can not be negative!')
        sys.exit()

    # Check floating point arguments
    if SINGLE_SQUARE <= 0:
        print('Single square size can not be negative!')
        sys.exit()

    # Check input path
    if not os.path.isdir(IMAGES_PATH_CAM1):
        print('The input cam1 path does not exist!')
        sys.exit()

    if len(os.listdir(IMAGES_PATH_CAM1)) == 0:
        print('The input cam1 folder is empty!')
        sys.exit()

    # Check input path
    if not os.path.isdir(IMAGES_PATH_CAM2):
        print('The input cam2 path does not exist!')
        sys.exit()

    if len(os.listdir(IMAGES_PATH_CAM2)) == 0:
        print('The input cam2 folder is empty!')
        sys.exit()

    # Check json format
    if not OUTPUT_PATH_CAM1.endswith('.json'):
        print('Output cam1 path does not ends up with .json format!')
        sys.exit()

    # Check json format
    if not OUTPUT_PATH_CAM2.endswith('.json'):
        print('Output cam2 path does not ends up with .json format!')
        sys.exit()

    print('All console arguments are valid!')


if __name__ == '__main__':
    # Console argument 
    parse_argumetns()
    validate_arguments()

    # Perform camera calibration
    (mtx1, dist1, rvecs1, tvecs1, objpoints1, imgpoints1, image_size_cam1, 
    mtx2, dist2, rvecs2, tvecs2, objpoints2, imgpoints2, image_size_cam2) = calibration_procedure()

    # Save result to json
    save_to_json(mtx1, dist1, rvecs1, tvecs1, image_size_cam1, objpoints1, imgpoints1, OUTPUT_PATH_CAM1)
    save_to_json(mtx2, dist2, rvecs2, tvecs2, image_size_cam2, objpoints2, imgpoints2, OUTPUT_PATH_CAM2)
