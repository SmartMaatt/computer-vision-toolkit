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
IMAGES_PATH = "..\..\stereo_datasets\cam2"
OUTPUT_PATH = "mono_calibration.json"


# Performs camera calibration process
def calibrate_camera():
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
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane [Corner sub pix]
    success_state_machine = [] # Success values of the calibration operation

    counter = 0
    print('Starting chessboard detection...')

    # Extracting paths of images stored in a given directory
    images = glob.glob(IMAGES_PATH + "\*.png")

    for fname in images:
        path = os.path.basename(fname).rsplit( ".", 1 )[ 0 ]
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, None)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of chessboard
        """
        counter += 1
        image_size = gray.shape[::-1]
        success_state_machine.append([counter - 1, ret])

        if ret == True:
            # Refining pixel coordinates for given 2d points.
            cornersSubPix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(cornersSubPix)

            print(f'Chessboard detection succeeded: [{counter}] {fname}')
        else:
            print(f'Chessboard detection failed: [{counter}] {fname}')


    print("Calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    print("Camera matrix:")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)
    print("\nRotation Vectors:")
    print(rvecs)
    print("\nTranslation Vectors:")
    print(tvecs)

    # Object points conversion
    for i, objpoint in enumerate(objpoints):
        objpoints[i] = np.asarray(objpoint, dtype=np.float32)

    # Image points conversion
    for i, imgpoint in enumerate(imgpoints):
        imgpoints[i] = np.asarray(imgpoint, dtype=np.float32)

    return (ret, mtx, dist, rvecs, tvecs, image_size, objpoints, imgpoints, success_state_machine)


# Save final parameter matrix to json
def save_to_json(mtx, dist, rvecs, tvecs, image_size, objpoints, imgpoints, success_state_machine):
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
        "imgPoints": imgpoints,
        "successStateMachine": success_state_machine
    }

    # Writing to json
    with open(OUTPUT_PATH, "w") as outfile:
        json.dump(json_result, outfile, indent=4, cls=NumpyArrayEncoder)


# Parses and interprets console line arguments
def parse_argumetns():
    global CHESSBOARD
    global SINGLE_SQUARE
    global IMAGES_PATH
    global OUTPUT_PATH

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--h', type=int, default=CHESSBOARD[0], metavar='Height', help='Height of checkerboard')
        parser.add_argument('--w', type=int, default=CHESSBOARD[1], metavar='Width', help='Width of checkerboard')
        parser.add_argument('--s', type=float, default=SINGLE_SQUARE, metavar='Square size', help='Real size of square')

        parser.add_argument('--i', type=str, default=IMAGES_PATH, metavar='Input path', help='Path to input image dataset')
        parser.add_argument('--o', type=str, default=OUTPUT_PATH, metavar='Output path', help='Path to output configuration json file')
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
    IMAGES_PATH = args.i
    print(f'Input image folder path: {IMAGES_PATH}')
    OUTPUT_PATH = args.o
    print(f'Output data json path: {OUTPUT_PATH}')
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
    if not os.path.isdir(IMAGES_PATH):
        print('The path specified does not exist!')
        sys.exit()

    # Check json format
    if not OUTPUT_PATH.endswith('.json'):
        print('Output path does not ends up with .json format!')
        sys.exit()

    print('All console arguments are valid!')


if __name__ == '__main__':
    # Console argument 
    parse_argumetns()
    validate_arguments()

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs, image_size, objpoints, imgpoints, success_state_machine = calibrate_camera()

    # Save result to json
    save_to_json(mtx, dist, rvecs, tvecs, image_size, objpoints, imgpoints, success_state_machine)
