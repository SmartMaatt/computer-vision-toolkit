
import os
import IO
import matplotlib
import numpy as np
import cv2 as cv

from math import tan
from enum import Enum
from matplotlib import pyplot as plt
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import open3d as o3d


class DispDirection(Enum):
    left_to_right = 0
    right_to_left = 1

class DispCriterium(Enum):
    argmax = 0
    argmin = 1

# >>> Disparity variables <<<
DISP_FILE = r"./disp_output_file.json"
use_saved_disp = True

max_disp = 64
window_size = (11, 11)
disp_direction = DispDirection.left_to_right
disp_criterium = DispCriterium.argmin

IMG_LEFT = r"./Car/left.png"
IMG_RIGHT = r"./Car/right.png"
# IMG_LEFT = r"./Cones/im2.png"
# IMG_RIGHT = r"./Cones/im6.png"
# IMG_LEFT = r"./Motocycle/im0.png"
# IMG_RIGHT = r"./Motocycle/im1.png"
#https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Motorcycle-perfect/
#https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/

# Dane w calib.txt datasetu - w tym przypadku dla bike1 i bike2
DOFFS = None
BASELINE = 0.6
F = None
FOV = 120




def calculate_focal_with_FOV(image_width, fov):
    return (image_width / (2 * tan(fov / 2)))

def calculate_disparity(img_left, img_right, max_disparity, window_size, direction, criterium):
    if use_saved_disp and os.path.isfile(DISP_FILE):
        disp = IO.read_disp_data(DISP_FILE)
        return disp

    else:
        if direction == DispDirection.right_to_left:
            disp = calculate_disparity_bm_from_right_to_left(img_left, img_right, max_disparity, window_size, criterium)
        elif direction == DispDirection.left_to_right:
            disp = calculate_disparity_bm_from_left_to_right(img_left, img_right, max_disparity, window_size, criterium)

        IO.save_disp_to_json(disp, DISP_FILE)
        return disp

def calculate_disparity_bm_from_right_to_left(img_left, img_right, max_disparity, window_size, criterium):
    height = np.shape(img_left)[0]
    width = np.shape(img_left)[1]
    window_height = window_size[0]
    window_width = window_size[1]
    half_window_height = window_height // 2
    half_window_width = window_width // 2
    disparity = np.zeros((height, width))

    for y in tqdm(range(half_window_height, height - half_window_height)):
        for x in range(width - half_window_width, half_window_width, -1):
            template = img_left[y - half_window_height: y + half_window_height, x - half_window_width: x + half_window_width]
            n_disparity = min(max_disparity, x - half_window_width)
            score = np.zeros(n_disparity)

            for offset in range(n_disparity, 0, -1):
                roi = img_right[y - half_window_height: y + half_window_height, x - half_window_width - offset: x + half_window_width - offset]
                score[offset - 1] = ssd(roi, template)

            if criterium == DispCriterium.argmax:
                disparity[y, x] = score.argmax()
            elif criterium == DispCriterium.argmin:
                disparity[y, x] = score.argmin()
    return disparity

def calculate_disparity_bm_from_left_to_right(img_left, img_right, max_disparity, window_size, criterium):
    height = np.shape(img_left)[0]
    width = np.shape(img_left)[1]
    window_height = window_size[0]
    window_width = window_size[1]
    half_window_height = window_height // 2
    half_window_width = window_width // 2
    disparity = np.zeros((height, width))

    for y in tqdm(range(half_window_height, height - half_window_height)):
        for x in range(half_window_width, width - half_window_width):
            template = img_right[y - half_window_height: y + half_window_height, x - half_window_width: x + half_window_width]
            n_disparity = min(max_disparity, width - x - half_window_width)
            score = np.zeros(n_disparity)

            for offset in range(n_disparity):
                roi = img_left[y - half_window_height: y + half_window_height, x - half_window_width + offset: x + half_window_width + offset]
                score[offset - 1] = ssd(template, roi)

            if criterium == DispCriterium.argmax:
                disparity[y, x] = score.argmax()
            elif criterium == DispCriterium.argmin:
                disparity[y, x] = score.argmin()
    return disparity

# Sum of square difference
def ssd(img_left, img_right):
    return np.sum((img_left - img_right) ** 2)






def disp_to_depth(disp, f, baseline, doffs):
    depth = np.zeros(shape=disp.shape)
    for i in range(len(disp)):
        for j in range(len(disp[i])):
            if (disp[i][j] == 0):
                depth[i][j] = (f * baseline) / 1
            else:
                depth[i][j] = (f * baseline) / (disp[i][j] + doffs)
    return depth

def depth_to_disp(depth, baseline, f, doffs):
    return baseline * f / depth - doffs
def depth_to_disp(depth, f, baseline, doffs):
    disp = np.zeros(shape=depth.shape)
    for i in range(len(depth)):
        for j in range(len(depth[i])):
            if (depth[i][j] == 0):
                disp[i][j] = (baseline * f) / 1
            else:
                disp[i][j] = (baseline * f) / (depth[i][j] - doffs)
    return disp

def compute_cx_cy(width,height):
    cx = width/2
    cy = height/2
    return (cx,cy)

def save_depth_to_ply(depth,fov):
    pcd = []
    height, width = depth.shape
    cx,cy = compute_cx_cy(width,height)
    for i in range(height):
        for j in range(width):
            z = depth[i][j]
            x = (j - cx) * z / fov #fx
            y = (i - cy) * z / fov #fy
            pcd.append([x, y, z])
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
    # Visualize:
    o3d.visualization.draw_geometries([pcd_o3d])





if __name__ == '__main__':
    img_left = cv.imread(IMG_LEFT)
    img_right = cv.imread(IMG_RIGHT)

    # Calculate disparity
    disp = calculate_disparity(img_left, img_right, max_disp, window_size, disp_direction, disp_criterium)

    # Plot and save disparity
    matplotlib.pyplot.imshow(disp)
    plt.show()
    plt.imsave("results/disparity.png", disp)
    cv.imwrite("results/disparity_raw.png", disp)

    # Calculate focal and depth
    F = calculate_focal_with_FOV(np.shape(img_left)[1], FOV)
    depth = disp_to_depth(disp, F, BASELINE, 0)

    imgPlot = matplotlib.pyplot.imshow(depth)
    plt.show()
    
    Fy = calculate_focal_with_FOV(np.shape(img_left)[0], FOV)
    save_depth_to_ply(depth,F) #fx fy???
    # Plot and save depth
    matplotlib.pyplot.imshow(depth)
    plt.show()
    plt.imsave("results/depth.png", depth)
    cv.imwrite("results/depth_raw.png", depth)

    # Save depth from file
    read_depth = IO.read_image_to_np_array("results/depth_raw.png")

    # Calculate disparity from read depth
    read_disp = depth_to_disp(read_depth, F, BASELINE, 0)

    # Plot read disparity
    matplotlib.pyplot.imshow(read_disp)
    plt.show()
