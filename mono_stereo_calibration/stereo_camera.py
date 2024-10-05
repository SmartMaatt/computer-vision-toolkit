import os
import sys
import argparse
import json
import time
from json import JSONEncoder

import cv2 as cv
import numpy as np

CALIB_CAM1_PATH = "multicam_calibration_cam1.json"
CALIB_CAM2_PATH = "multicam_calibration_cam2.json"

STEREO_CALIB_PATH = "calibration_stereo.json"
ALPHA_SCALE = 0
USE_SAVED_STEREO_CALIB = False
REMAP_TEST = False

IMAGE_CAM1_PATH = ""
IMAGE_CAM2_PATH = ""
OUTPUT_PATH = 'rectified.png'


def stereo_calibrate(objPoints, imgPoints1, imgPoints2, imgSize, mtx1, mtx2, dist1, dist2):
    print(f'\nStereo calibration')

    # Read stereo calibration from file
    if USE_SAVED_STEREO_CALIB and os.path.isfile(STEREO_CALIB_PATH):
        smtx1, sdist1, smtx2, sdist2, R, T, E, F, baseline, fov_cam1, fov_cam2 = read_stereo_camera_calibration(STEREO_CALIB_PATH)
        
        print(f'Baseline: {baseline}cm')
        print(f'FOV of cam1: {fov_cam1}')
        print(f'FOV of cam2: {fov_cam2}')
        return (smtx1, sdist1, smtx2, sdist2, R, T, E, F, baseline, fov_cam1, fov_cam2)

    # Basic stereo calibration
    else:
        flags = cv.CALIB_FIX_INTRINSIC
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        retval, smtx1, sdist1, smtx2, sdist2, R, T, E, F = cv.stereoCalibrate(objPoints, imgPoints1, imgPoints2, mtx1, dist1, mtx2, dist2, imgSize, criteria=criteria, flags=flags)    
        baseline = round(np.linalg.norm(T)*0.1, 2)
        fov_cam1 = calculate_fov(smtx1, image_size1)
        fov_cam2 = calculate_fov(smtx2, image_size2)

        print(f'Baseline: {baseline}cm')
        print(f'FOV of cam1: {fov_cam1}')
        print(f'FOV of cam2: {fov_cam2}')
        return (smtx1, sdist1, smtx2, sdist2, R, T, E, F, baseline, fov_cam1, fov_cam2)


def calculate_fov(smtx, imgSize):
    fx = smtx[0][0]
    fy = smtx[1][1]
    width, height = imgSize

    fovW = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    fovH = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
    return fovW, fovH


def rectification(smtx1, dist1, smtx2, dist2, imgSize, R, T):
    print('\nRectification')
    print(f'Alpha scale: {ALPHA_SCALE}')

    rectify_scale = ALPHA_SCALE
    R1, R2, P1, P2, Q, sroi1, sroi2 = cv.stereoRectify(smtx1, dist1, smtx2, dist2, imgSize, R, T, alpha=rectify_scale)
    map1_cam1, map2_cam1 = cv.initUndistortRectifyMap(smtx1, dist1, R1, P1, imgSize, m1type=cv.CV_16SC2)
    map1_cam2, map2_cam2 = cv.initUndistortRectifyMap(smtx2, dist2, R2, P2, imgSize, m1type=cv.CV_16SC2)
    return map1_cam1, map2_cam1, map1_cam2, map2_cam2, sroi1, sroi2


def remapAndShowRect(map1_cam1, map2_cam1, map1_cam2, map2_cam2, image_cam1, image_cam2, sroi1, sroi2):
    print('\nRemap and showrect')
    if REMAP_TEST:
        remap_test(map1_cam1, map2_cam1, map1_cam2, map2_cam2, image_cam1, image_cam2)
    
    img1 = cv.imread(image_cam1)
    img2 = cv.imread(image_cam2)

    remap_method = cv.INTER_LINEAR
    imgAfterRect1 = cv.remap(img1, map1_cam1, map2_cam1, remap_method)
    imgAfterRect2 = cv.remap(img2, map1_cam2, map2_cam2, remap_method)    

    cv.rectangle(imgAfterRect1, (sroi1[0], sroi1[1]), (sroi1[2], sroi1[3]), (0, 255, 0), 2)
    cv.rectangle(imgAfterRect2, (sroi2[0], sroi2[1]), (sroi2[2], sroi2[3]), (0, 255, 0), 2)

    catImgsAfterRect = np.concatenate((imgAfterRect1, imgAfterRect2), axis=1)
    for i in range(10, np.shape(catImgsAfterRect)[0], 50):
        cv.line(catImgsAfterRect, (0, i), (np.shape(catImgsAfterRect)[1], i), (0, 0, 0), 1)

    cv.imwrite(OUTPUT_PATH, catImgsAfterRect)


def remap_test(map1_cam1, map2_cam1, map1_cam2, map2_cam2, image_cam1, image_cam2):
    print("Remap interpolation test")
    img1 = cv.imread(image_cam1)
    img2 = cv.imread(image_cam2)
    timer = time.time()

    # INTER_NEAREST
    imgAfterRect1 = cv.remap(img1, map1_cam1, map2_cam1, cv.INTER_NEAREST)
    imgAfterRect2 = cv.remap(img2, map1_cam2, map2_cam2, cv.INTER_NEAREST)
    print(f'INTER_NEAREST: {round(time.time() - timer, 4)}s')
    timer = time.time()

    # INTER_LINEAR
    imgAfterRect1 = cv.remap(img1, map1_cam1, map2_cam1, cv.INTER_LINEAR)
    imgAfterRect2 = cv.remap(img2, map1_cam2, map2_cam2, cv.INTER_LINEAR)
    print(f'INTER_LINEAR: {round(time.time() - timer, 4)}s')
    timer = time.time()

    # INTER_CUBIC
    imgAfterRect1 = cv.remap(img1, map1_cam1, map2_cam1, cv.INTER_CUBIC)
    imgAfterRect2 = cv.remap(img2, map1_cam2, map2_cam2, cv.INTER_CUBIC)
    print(f'INTER_CUBIC: {round(time.time() - timer, 4)}s')
    timer = time.time()

    # INTER_AREA
    imgAfterRect1 = cv.remap(img1, map1_cam1, map2_cam1, cv.INTER_AREA)
    imgAfterRect2 = cv.remap(img2, map1_cam2, map2_cam2, cv.INTER_AREA)
    print(f'INTER_AREA: {round(time.time() - timer, 4)}s')
    timer = time.time()

    # INTER_LANCZOS4
    imgAfterRect1 = cv.remap(img1, map1_cam1, map2_cam1, cv.INTER_LANCZOS4)
    imgAfterRect2 = cv.remap(img2, map1_cam2, map2_cam2, cv.INTER_LANCZOS4)
    print(f'INTER_LANCZOS4: {round(time.time() - timer, 4)}s')


# Reads stereo camera calibration data from json file
def read_stereo_camera_calibration(calibration_path):
    # Read JSON file
    print(f'Reading stereo calibration file: {calibration_path}')

    with open(fr'{calibration_path}', 'r') as f:
      calibration_data = json.load(f)

    smtx1 = np.array(calibration_data['smtx1'])
    sdist1 = np.array(calibration_data['sdist1'])
    smtx2 = np.array(calibration_data['smtx2'])
    sdist2 = np.array(calibration_data['sdist2'])
    R = np.array(calibration_data['R'])
    T = np.array(calibration_data['T'])
    E = np.array(calibration_data['E'])
    F = np.array(calibration_data['F'])
    baseline = calibration_data['Baseline'],
    fov_cam1 = calibration_data['Fov_cam1'],
    fov_cam2 = calibration_data['Fov_cam2']
    
    return (smtx1, sdist1, smtx2, sdist2, R, T, E, F, baseline, fov_cam1, fov_cam2)


# Reads camera calibration data from json file
def read_camera_calibration_data(calibration_path):
    # Read JSON file
    print(f'Reading calibration file: {calibration_path}')
    with open(fr'{calibration_path}', 'r') as f:
      calibration_data = json.load(f)

    mtx = np.array(calibration_data['mtx'])
    dist = np.array(calibration_data['dist'])
    rvecs = np.array(calibration_data['rvecs'])
    tvecs = np.array(calibration_data['tvecs'])
    image_size = calibration_data['imageSize']
    objpoints = calibration_data['objPoints']
    imgpoints = calibration_data['imgPoints']

    for i, objpoint in enumerate(objpoints):
            objpoints[i] = np.asarray(objpoint, dtype=np.float32)
    for i, imgpoint in enumerate(imgpoints):
        imgpoints[i] = np.asarray(imgpoint, dtype=np.float32)

    return (mtx, dist, rvecs, tvecs, image_size, objpoints, imgpoints)


# Save final parameter matrix to json
def save_stereo_to_json(smtx1, sdist1, smtx2, sdist2, R, T, E, F, baseline, fov_cam1, fov_cam2):
    # >>> Saving result to json <<<
    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    json_data = {
        'smtx1': smtx1,
        'sdist1': sdist1,
        'smtx2': smtx2,
        'sdist2': sdist2,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'Baseline': baseline,
        'Fov_cam1': fov_cam1,
        'Fov_cam2': fov_cam2
    }

    # Writing to json
    with open(STEREO_CALIB_PATH, "w") as outfile:
        json.dump(json_data, outfile, indent=4, cls=NumpyArrayEncoder)


# Parses and interprets console line arguments
def parse_argumetns():
    global CALIB_CAM1_PATH
    global CALIB_CAM2_PATH

    global STEREO_CALIB_PATH
    global ALPHA_SCALE
    global USE_SAVED_STEREO_CALIB
    global REMAP_TEST   

    global IMAGE_CAM1_PATH
    global IMAGE_CAM2_PATH
    global OUTPUT_PATH
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--c1', type=str, default=CALIB_CAM1_PATH, help='Input path to camera one calibration data')
        parser.add_argument('--c2', type=str, default=CALIB_CAM2_PATH, help='Input path to camera two calibration data')

        parser.add_argument('--s', type=str, default=STEREO_CALIB_PATH, help='Output path to stereo calibration data')
        parser.add_argument('--a', type=str, default=ALPHA_SCALE, help='Rectification alpha scale parameter')
        parser.add_argument('--u', type=bool, default=USE_SAVED_STEREO_CALIB, help='Use saved stereo calibration data')
        parser.add_argument('--r', type=bool, default=REMAP_TEST, help='Test different remap interpolations')

        parser.add_argument('--i1', type=str, default=IMAGE_CAM1_PATH, help='Input camera one test image')
        parser.add_argument('--i2', type=str, default=IMAGE_CAM2_PATH, help='Input camera two test image')
        parser.add_argument('--o', type=str, default=OUTPUT_PATH, help='Output image file path')
    except argparse.ArgumentError as ex:
        print(ex)
        sys.exit()

    # Save arguments to global variables
    args = parser.parse_args()
    print('>>> Inserted arguments <<<')
    CALIB_CAM1_PATH = args.c1
    print(f'Camera one calibration path: {CALIB_CAM1_PATH}')
    CALIB_CAM2_PATH = args.c2
    print(f'Camera two calibration path: {CALIB_CAM2_PATH}')
    STEREO_CALIB_PATH = args.s
    print(f'Stereo calibration path: {STEREO_CALIB_PATH}')
    ALPHA_SCALE = args.a
    print(f'Alpha scale parameter: {ALPHA_SCALE}')
    USE_SAVED_STEREO_CALIB = args.u
    print(f'Use saved stereo calibration data: {USE_SAVED_STEREO_CALIB}')
    REMAP_TEST = args.r
    print(f'Test different remap interpolations: {REMAP_TEST}')
    IMAGE_CAM1_PATH = args.i1
    print(f'Camera one test image: {IMAGE_CAM1_PATH}')
    IMAGE_CAM2_PATH = args.i2
    print(f'Camera two test image: {IMAGE_CAM2_PATH}')
    OUTPUT_PATH = args.o
    print(f'Output image path: {OUTPUT_PATH}')
    print('')


# Validates console line arguments
def validate_arguments():
    # Check json format
    if not CALIB_CAM1_PATH.endswith('.json'):
        print('Camera one input path does not ends up with .json format!')
        sys.exit()

    if not CALIB_CAM2_PATH.endswith('.json'):
        print('Camera two input path does not ends up with .json format!')
        sys.exit()

    if not STEREO_CALIB_PATH.endswith('.json'):
        print('Stereo input path does not ends up with .json format!')
        sys.exit()

    if not IMAGE_CAM1_PATH.endswith('.png'):
        print('Camera one test image path does not ends up with .png format!')
        sys.exit()

    if not IMAGE_CAM2_PATH.endswith('.png'):
        print('Camera two test image path does not ends up with .png format!')
        sys.exit()

    print('All console arguments are valid!')


if __name__ == "__main__":
    # Console argument 
    parse_argumetns()
    validate_arguments()

    # Cameras calibration
    mtx1, dist1, rvecs1, tvecs1, image_size1, objpoints1, imgpoints1 = read_camera_calibration_data(CALIB_CAM1_PATH)
    mtx2, dist2, rvecs2, tvecs2, image_size2, objpoints2, imgpoints2 = read_camera_calibration_data(CALIB_CAM2_PATH)

    # Stereo calibration
    smtx1, sdist1, smtx2, sdist2, R, T, E, F, baseline, fov_cam1, fov_cam2 = stereo_calibrate(objpoints1, imgpoints1, imgpoints2, image_size1, mtx1, mtx2, dist1, dist2)
    save_stereo_to_json(smtx1, sdist1, smtx2, sdist2, R, T, E, F, baseline, fov_cam1, fov_cam2)

    map1_cam1, map2_cam1, map1_cam2, map2_cam2, sroi1, sroi2 = rectification(smtx1, sdist1, smtx2, sdist2, image_size1, R, T)
    remapAndShowRect(map1_cam1, map2_cam1, map1_cam2, map2_cam2, IMAGE_CAM1_PATH, IMAGE_CAM2_PATH, sroi1, sroi2)
