import cv2 as cv
import numpy as np
import json
import sys
import os
import argparse

# Defining the dimensions of chessboard
INPUT_PATH = "..\..\stero_datasets\cam2\15.png"
CONFIGURATION_PATH = "mono_calibration_cam1.json"
OUTPUT_PATH = "."


# Removing distortion
def remove_distortion(mtx, dist):
    print('>>> Removing distortion of image - initUndistortRectifyMap i remap method <<<')

    # load image
    img = cv.imread(fr'{INPUT_PATH}')

    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort + remap
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image by R.egion Of I.nterest
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    img = cv.resize(img, (w, h), interpolation = cv.INTER_AREA)

    cv.imwrite(fr'{OUTPUT_PATH}\undistort_remap_with_distortion.png', img)
    print('Saved pre processing photo to "undistort_remap_with_distortion.png"')
    cv.imwrite(fr'{OUTPUT_PATH}\undistort_remap_no_distortion.png', dst)
    print('Saved post processing photo to "undistort_remap_no_distortion.png"')

    vertical_image_grid = cv.vconcat([img, dst])
    h, w = vertical_image_grid.shape[:2]
    vertical_image_grid = cv.resize(vertical_image_grid, (int(w*0.7), int(h*0.7)), interpolation = cv.INTER_AREA)
    cv.imshow('Processing comparation', vertical_image_grid)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Reads camera calibration data from json file
def read_camera_calibration_data():
    # Read JSON file
    print(f'Reading calibration file: {CONFIGURATION_PATH}')

    with open(fr'{CONFIGURATION_PATH}', 'r') as f:
      calibration_data = json.load(f)

    mtx = np.array(calibration_data['mtx'])
    dist = np.array(calibration_data['dist'])

    print('')
    return (mtx, dist)


# Parses and interprets console line arguments
def parse_argumetns():
    global INPUT_PATH
    global CONFIGURATION_PATH
    global OUTPUT_PATH
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--i', type=str, default=INPUT_PATH, metavar='Image input path', help='Path to input image')
        parser.add_argument('--c', type=str, default=CONFIGURATION_PATH, metavar='Calibration file path', help='Path to camera configuration json file')
        parser.add_argument('--o', type=str, default=OUTPUT_PATH, metavar='Output path', help='Directory path of output')
    except argparse.ArgumentError as ex:
        print(ex)
        sys.exit()

    # Save arguments to global variables
    args = parser.parse_args()

    print('>>> Inserted arguments <<<')
    INPUT_PATH = args.i
    print(f'Input image folder path: {INPUT_PATH}')
    CONFIGURATION_PATH = args.c
    print(f'Configuration file path: {CONFIGURATION_PATH}')
    OUTPUT_PATH = args.o
    print(f'Output folder path: {OUTPUT_PATH}')
    print('')


# Validates console line arguments
def validate_arguments():
    # Check input format
    if not INPUT_PATH.endswith('.png'):
        print('Input path does not ends up with .png format!')
        sys.exit()

    # Check configuation json format
    if not CONFIGURATION_PATH.endswith('.json'):
        print('Configuration path does not ends up with .json format!')
        sys.exit()

    if not os.path.isdir(OUTPUT_PATH):
        print(f'No directory with path {OUTPUT_PATH}')
        sys.exit()

    print('All console arguments are valid!')


if __name__ == '__main__':
    # Console argument management
    parse_argumetns()
    validate_arguments()

    # Read calibration data
    mtx, dist = read_camera_calibration_data()

    # Removing distortion
    remove_distortion(mtx, dist)