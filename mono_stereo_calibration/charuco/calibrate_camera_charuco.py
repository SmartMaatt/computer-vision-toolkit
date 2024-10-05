import os
import sys
import glob
import argparse
import json
from json import JSONEncoder

import cv2 as cv
import numpy as np

# Defining the dimensions of chessboard
CHESSBOARD = (8,11)
SQUARE_LENGTH_MM = 44.0
MARKER_LENGTH_MM = 34.0
MIN_CORNERS_COUNT = 3
DISPLAY_VALID = True

# https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html#ga4e13135a118f497c6172311d601ce00d
ARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50)
BOARD = cv.aruco.CharucoBoard(CHESSBOARD, SQUARE_LENGTH_MM, MARKER_LENGTH_MM, ARUCO_DICT)
IMAGES_PATH = r""
OUTPUT_PATH = "mono_calibration_charuco.json"
PHOTOS_INFO_PATH = "used_photos_mono_charuco.json"


# Performs camera calibration process
def calibrate_camera():
    image_size = None

    # Arrays to store charuco corners and ids
    counter = 0
    all_corners = []
    all_ids = []
    all_objpoints = []  # 3d point in real world space
    all_imgpoints = []  # 2d points in image plane [Corner sub pix]

    used_photos = []
    failed_photos = []
    print('Starting chessboard detection...')

    # Extracting paths of images stored in a given directory
    paths = os.path.join(IMAGES_PATH, "*.jpg")
    images = glob.glob(paths)

    for fname in images:
        path = os.path.basename(fname).rsplit( ".", 1 )[ 0 ]
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, ARUCO_DICT)
        counter += 1
        image_size = gray.shape[::-1]
        
        if DISPLAY_VALID:
            img = cv.aruco.drawDetectedMarkers(img, corners)

        if len(corners) > 0:
            ret, c_corners, c_ids = cv.aruco.interpolateCornersCharuco(corners, ids, gray, BOARD)

            # ret is the number of detected corners
            if ret > MIN_CORNERS_COUNT:
                objpoints, imgpoints = cv.aruco.getBoardObjectAndImagePoints(BOARD, c_corners, c_ids)
                all_corners.append(c_corners)
                all_ids.append(c_ids)
                all_objpoints.append(objpoints)
                all_imgpoints.append(imgpoints)
                print(f'Chessboard detection succeeded:\t[{counter}] {fname} [Found corners: {ret}/{((CHESSBOARD[0]-1) * (CHESSBOARD[1]-1))}]')

                used_photos.append(fname)

                if DISPLAY_VALID:
                    img = cv.aruco.drawDetectedCornersCharuco(img, c_corners, c_ids)
                    scale = 0.8
                    width = int(img.shape[1] * scale)
                    height = int(img.shape[0] * scale)
                    img = cv.resize(img, (width, height))
                    cv.imshow('frame',img)
                    cv.waitKey(0)
            else:
                print(f'Required minimum {MIN_CORNERS_COUNT} corners:\t[{counter}] {fname} [Found corners: {ret}/{((CHESSBOARD[0]-1) * (CHESSBOARD[1]-1))}]')
                failed_photos.append(fname)

        else:
            print(f'Chessboard detection failed:\t[{counter}] {fname}')
            failed_photos.append(fname)

    print("Calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv.aruco.calibrateCameraCharuco(all_corners, all_ids, BOARD, image_size, None, None)
    print("Camera matrix:")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)
    # print("\nRotation Vectors:")
    # print(rvecs)
    # print("\nTranslation Vectors:")
    # print(tvecs)

    # Saving used photos info
    save_photos_info(used_photos, failed_photos, PHOTOS_INFO_PATH)

    return (ret, mtx, dist, rvecs, tvecs, image_size, all_corners, all_ids, all_objpoints, all_imgpoints)


# Save final parameter matrix to json
def save_to_json(mtx, dist, rvecs, tvecs, image_size, all_corners, all_ids, all_objpoints, all_imgpoints):
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
        "allCorners" : all_corners,
        "allIds" : all_ids,
        "objPoints": all_objpoints,
        "imgPoints": all_imgpoints
    }

    # Writing to json
    with open(OUTPUT_PATH, "w") as outfile:
        json.dump(json_result, outfile, indent=4, cls=NumpyArrayEncoder)


def save_photos_info(used_photos, failed_photos, output_path):
    json_result = {
        "usedPhotos": used_photos,
        "failedPhotos": failed_photos
    }

    # Writing to json
    with open(output_path, "w") as outfile:
        json.dump(json_result, outfile, indent=4)



# Parses and interprets console line arguments
def parse_argumetns():
    global CHESSBOARD
    global SQUARE_LENGTH_MM
    global MARKER_LENGTH_MM
    global IMAGES_PATH
    global OUTPUT_PATH

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--h', type=int, default=CHESSBOARD[1], metavar='Height', help='Height of checkerboard')
        parser.add_argument('--w', type=int, default=CHESSBOARD[0], metavar='Width', help='Width of checkerboard')
        parser.add_argument('--s', type=float, default=SQUARE_LENGTH_MM, metavar='Square size', help='Real size of square')
        parser.add_argument('--q', type=float, default=MARKER_LENGTH_MM, metavar='Marker size', help='Real size of marker')

        parser.add_argument('--i', type=str, default=IMAGES_PATH, metavar='Input path', help='Path to input image dataset')
        parser.add_argument('--o', type=str, default=OUTPUT_PATH, metavar='Output path', help='Path to output configuration json file')
    except argparse.ArgumentError as ex:
        print(ex)
        sys.exit()

    # Save arguments to global variables
    args = parser.parse_args()

    print('>>> Inserted arguments <<<')
    CHESSBOARD = (args.w, args.h)
    print(f'Chessboard size: ({CHESSBOARD[0]}, {CHESSBOARD[1]})')
    SQUARE_LENGTH_MM = args.s
    print(f'Real square size: {SQUARE_LENGTH_MM}mm')
    MARKER_LENGTH_MM = args.q
    print(f'Real marker size: {MARKER_LENGTH_MM}mm')
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
    if SQUARE_LENGTH_MM <= 0:
        print('Single square size can not be negative!')
        sys.exit()

    if MARKER_LENGTH_MM <= 0:
        print('Single marker size can not be negative!')
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
    ret, mtx, dist, rvecs, tvecs, image_size, all_corners, all_ids, all_objpoints, all_imgpoints = calibrate_camera()

    # Save result to json
    save_to_json(mtx, dist, rvecs, tvecs, image_size, all_corners, all_ids, all_objpoints, all_imgpoints)
