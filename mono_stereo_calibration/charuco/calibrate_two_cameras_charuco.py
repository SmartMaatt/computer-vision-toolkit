import os
import sys
import glob
import argparse
import json
from json import JSONEncoder

import cv2 as cv
import numpy as np

# Defining the dimensions of chessboard
CHESSBOARD = (8, 11)
SQUARE_LENGTH_MM = 44.0
MARKER_LENGTH_MM = 34.0
MIN_CORNERS_COUNT = 1
DISPLAY_VALID = False

ARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50)
BOARD = cv.aruco.CharucoBoard(CHESSBOARD, SQUARE_LENGTH_MM, MARKER_LENGTH_MM, ARUCO_DICT)
IMAGES_PATH_CAM1 = r""
IMAGES_PATH_CAM2 = r""
OUTPUT_PATH_CAM1 = "multicam_calibration_cam1_charuco.json"
OUTPUT_PATH_CAM2 = "multicam_calibration_cam2_charuco.json"
PHOTOS_INFO_PATH = "multicam_photos_used_charuco.json"


def filter_matched_points(corners_left, ids_left, corners_right, ids_right):
    # Create lists to store filtered points and identifiers
    filtered_corners_left = []
    filtered_corners_right = []
    filtered_ids = []

    # Iterate through all identifiers and corners in the left set
    for i, id in enumerate(ids_left):
        # Check if the identifier is also present in the right set
        if id in ids_right:
            # Add corners and identifiers to filtered lists if present in both sets
            idx_right = np.where(ids_right == id)[0][0]
            filtered_corners_left.append(corners_left[i])
            filtered_corners_right.append(corners_right[idx_right])
            filtered_ids.append(id)

    return np.array(filtered_corners_left), np.array(filtered_corners_right), np.array(filtered_ids)


# Performs camera calibration procedure
def calibration_procedure():
    # Arrays to store charuco corners and ids
    counter = 0
    all_corners_cam1 = []
    all_corners_cam2 = []
    all_ids_cam1 = []
    all_ids_cam2 = []
    all_objpoints_cam1 =[]
    all_objpoints_cam2 =[]
    all_imgpoints_cam1 = []
    all_imgpoints_cam2 = []

    used_photos = []
    failed_photos = []

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros(((CHESSBOARD[0] - 1) * (CHESSBOARD[1] - 1), 3), np.float32)                                   # Define [70][3] array filled with zeros
    objp[:,:2] = np.mgrid[0:(CHESSBOARD[0] - 1), 0:(CHESSBOARD[1] - 1)].T.reshape(-1, 2) * SQUARE_LENGTH_MM       # Fills indexies to elements

    counter = 0
    accepted_counter = 0
    print('Starting chessboard detection...')

    # Extracting paths of images stored in a given directory
    images_cam1 = glob.glob(IMAGES_PATH_CAM1 + "\*.bmp")
    images_cam2 = glob.glob(IMAGES_PATH_CAM2 + "\*.bmp")

    # Removing uncommon files
    images_cam1, images_cam2 = remove_uncommon_images(images_cam1, images_cam2)
    loop_counter = len(images_cam1)

    for i in range(loop_counter):
        img_cam1 = cv.imread(images_cam1[i])
        img_cam2 = cv.imread(images_cam2[i])

        gray_cam1 = cv.cvtColor(img_cam1, cv.COLOR_BGR2GRAY)
        gray_cam2 = cv.cvtColor(img_cam2, cv.COLOR_BGR2GRAY)

        corners_cam1, ids_cam1, rejectedImgPoints_cam1 = cv.aruco.detectMarkers(gray_cam1, ARUCO_DICT)
        corners_cam2, ids_cam2, rejectedImgPoints_cam2 = cv.aruco.detectMarkers(gray_cam2, ARUCO_DICT)

        counter += 1
        image_size_cam1 = gray_cam1.shape[::-1]
        image_size_cam2 = gray_cam2.shape[::-1]

        if image_size_cam1 != image_size_cam2:
            print('Couple rejected due to diffrent image sizes')
            continue
        
        if len(corners_cam1) > 0 and len(corners_cam2):
            ret_cam1, c_corners_cam1, c_ids_cam1 = cv.aruco.interpolateCornersCharuco(corners_cam1, ids_cam1, gray_cam1, BOARD)
            ret_cam2, c_corners_cam2, c_ids_cam2 = cv.aruco.interpolateCornersCharuco(corners_cam2, ids_cam2, gray_cam2, BOARD)

            # ret is the number of detected corners
            if ret_cam1 >= MIN_CORNERS_COUNT and ret_cam2 >= MIN_CORNERS_COUNT:
                # Charuco parameters
                all_corners_cam1.append(c_corners_cam1)
                all_corners_cam2.append(c_corners_cam2)
                all_ids_cam1.append(c_ids_cam1)
                all_ids_cam2.append(c_ids_cam2)
                accepted_counter+=1

                # Obj/Img points
                filtered_corners_cam1, filtered_corners_cam2, filtered_ids = filter_matched_points(c_corners_cam1, c_ids_cam1, c_corners_cam2, c_ids_cam2)
                objpoints = np.array([objp[i] for i in filtered_ids])

                all_objpoints_cam1.append(objpoints)
                all_objpoints_cam2.append(objpoints)
                all_imgpoints_cam1.append(filtered_corners_cam1)
                all_imgpoints_cam2.append(filtered_corners_cam2)

                # Image paths
                used_photos.append(images_cam1[i])
                used_photos.append(images_cam2[i])

                print(f'Chessboard detection succeeded:\t[{counter}] {images_cam1[i]} [Found corners: {ret_cam1}/{((CHESSBOARD[0]-1) * (CHESSBOARD[1]-1))}]')
                print(f'Chessboard detection succeeded:\t[{counter}] {images_cam2[i]} [Found corners: {ret_cam2}/{((CHESSBOARD[0]-1) * (CHESSBOARD[1]-1))}]')

                if DISPLAY_VALID:
                    img_cam1 = cv.aruco.drawDetectedMarkers(img_cam1, corners_cam1, ids_cam1)
                    img_cam2 = cv.aruco.drawDetectedMarkers(img_cam2, corners_cam2, ids_cam2)

                    scale = 0.2
                    width = int(image_size_cam1[0] * scale)
                    height = int(image_size_cam1[1] * scale)
                    img_cam1 = cv.resize(img_cam1, (width, height))
                    img_cam2 = cv.resize(img_cam2, (width, height))

                    cv.imshow('Image cam1', img_cam1)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

                    cv.imshow('Image cam2', img_cam2)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
            else:
                print(f'Required minimum {MIN_CORNERS_COUNT} corners:\t[{counter}] {images_cam1[i]} [Found corners: {ret_cam1}/{((CHESSBOARD[0]-1) * (CHESSBOARD[1]-1))}]')
                print(f'Required minimum {MIN_CORNERS_COUNT} corners:\t[{counter}] {images_cam2[i]} [Found corners: {ret_cam2}/{((CHESSBOARD[0]-1) * (CHESSBOARD[1]-1))}]')
                failed_photos.append(images_cam1[i])
                failed_photos.append(images_cam2[i])
        else:
            print(f'Chessboard detection failed:\t[{counter}] {images_cam1[i]}')
            print(f'Chessboard detection failed:\t[{counter}] {images_cam2[i]}')
            print('Couple rejected')
            failed_photos.append(images_cam1[i])
            failed_photos.append(images_cam2[i])

    save_photos_info(used_photos, failed_photos, PHOTOS_INFO_PATH)

    if accepted_counter == 0:
        print("No images accepted, calibration aborted")
        sys.exit()

    print("\n>>> Calibrating camera one... <<<")
    mtx1, dist1, rvecs1, tvecs1 = calibrate_camera(all_corners_cam1, all_ids_cam1, BOARD, image_size_cam1)

    print("\n>>> Calibrating camera two... <<<")
    mtx2, dist2, rvecs2, tvecs2 = calibrate_camera(all_corners_cam2, all_ids_cam2, BOARD, image_size_cam2)

    return (mtx1, dist1, rvecs1, tvecs1, all_corners_cam1, all_ids_cam1, all_objpoints_cam1, all_imgpoints_cam1, image_size_cam1, mtx2, dist2, rvecs2, tvecs2, all_corners_cam2, all_ids_cam2, all_objpoints_cam2, all_imgpoints_cam2, image_size_cam2)


# Calibrates camera based on input parametes
def calibrate_camera(all_corners, all_ids, BOARD, image_size):
    ret, mtx, dist, rvecs, tvecs = cv.aruco.calibrateCameraCharuco(all_corners, all_ids, BOARD, image_size, None, None)

    if ret:
        print("Camera calibration succeeded")
    else:
        print("Camera calibration failed")
        sys.exit()

    print("Camera matrix:")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)

    return (mtx, dist, rvecs, tvecs)


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
def save_to_json(mtx, dist, rvecs, tvecs, image_size, all_corners, all_ids, all_objpoints, all_imgpoints, output_path):
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
        "allCorners": all_corners,
        "allIds": all_ids,
        "objPoints" : all_objpoints,
        "imgPoints" : all_imgpoints    
    }

    # Writing to json
    with open(output_path, "w") as outfile:
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
    global MIN_CORNERS_COUNT

    global IMAGES_PATH_CAM1
    global IMAGES_PATH_CAM2
    global OUTPUT_PATH_CAM1
    global OUTPUT_PATH_CAM2

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--w', type=int, default=CHESSBOARD[0], metavar='Width', help='Width of checkerboard')
        parser.add_argument('--h', type=int, default=CHESSBOARD[1], metavar='Height', help='Height of checkerboard')
        parser.add_argument('--s', type=float, default=SQUARE_LENGTH_MM, metavar='Square size', help='Real size of square')
        parser.add_argument('--q', type=float, default=MARKER_LENGTH_MM, metavar='Marker size', help='Real size of marker')
        parser.add_argument('--m', type=float, default=MIN_CORNERS_COUNT, metavar='Minimum corner count', help='Number of minimum corners found to accept an image')

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
    CHESSBOARD = (args.w, args.h)
    print(f'Chessboard size: ({CHESSBOARD[0]}, {CHESSBOARD[1]})')
    SQUARE_LENGTH_MM = args.s
    print(f'Real square size: {SQUARE_LENGTH_MM}mm')
    MARKER_LENGTH_MM = args.q
    print(f'Real marker size: {MARKER_LENGTH_MM}mm')
    MIN_CORNERS_COUNT = args.m
    print(f'Minimum corners count: {MIN_CORNERS_COUNT}')

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
    if SQUARE_LENGTH_MM <= 0:
        print('Single square size can not be negative or zero!')
        sys.exit()

    if MARKER_LENGTH_MM <= 0:
        print('Single marker size can not be negative or zero!')
        sys.exit()

    if MIN_CORNERS_COUNT <= 0:
        print('Minimum corners found can not be negative or zero!')
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
    (mtx1, dist1, rvecs1, tvecs1, all_corners_1, all_ids_1, all_objpoints_1, all_imgpoints_1, image_size_1, 
     mtx2, dist2, rvecs2, tvecs2, all_corners_2, all_ids_2, all_objpoints_2, all_imgpoints_2, image_size_2) = calibration_procedure()

    # Save result to json
    save_to_json(mtx1, dist1, rvecs1, tvecs1, image_size_1, all_corners_1, all_ids_1, all_objpoints_1, all_imgpoints_1, OUTPUT_PATH_CAM1)
    save_to_json(mtx2, dist2, rvecs2, tvecs2, image_size_2, all_corners_2, all_ids_2, all_objpoints_2, all_imgpoints_2, OUTPUT_PATH_CAM2)
