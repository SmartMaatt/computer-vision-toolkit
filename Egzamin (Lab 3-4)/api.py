import math
import multiprocessing as mlp
import numpy as np
import cv2 as cv

from math import tan
from enum import Enum
from matplotlib import pyplot as plt
from tqdm import tqdm
import open3d as o3d


class DispDirection(Enum):
    left_to_right = 0
    right_to_left = 1


class DispCriterium(Enum):
    argmax = 0
    argmin = 1


# >>> Global variables <<<
DISP_FILE = r"./disp_output_file_Egz34.json"
USE_SAVED_DISP = False

# >>> General purpose methods <<<
def clamp(min, max, value):
    if value < min:
        value = min
    elif value > max:
        value = max
    return value


def calculate_focal_with_FOV(image_width, fov):
    return (image_width / (2 * tan(fov / 2)))


def calculate_baseline_with_T_matrix(T):
    return round(np.linalg.norm(T) * 0.1, 2)


def calculate_fovX_with_focalX(image_height, fx):
    return 2 * math.atan(image_height / (2 * fx))


def calculate_fovY_with_focalY(image_width, fy):
    return 2 * math.atan(image_width / (2 * fy))


def calculate_cx_cy_with_image_size(width, height):
    cx = width / 2
    cy = height / 2
    return (cx, cy)


def image_to_bgra(image):
    array = np.frombuffer(image, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.shape[0], image.shape[1], 4))
    return array


def image_to_rgb(image):
    array = image_to_bgra(image)
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


# >>> Disparity methods <<<
def calculate_disparity_with_SGBM(img_left, img_right, max_disparity=64, window_size=11):
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=max_disparity,
        blockSize=window_size,
        P1=3 * 4 * window_size ** 2,
        P2=3 * 32 * window_size ** 2,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv.StereoSGBM_MODE_HH)
    disp = stereo.compute(img_left, img_right).astype('float32') / 16
    return disp


def calculate_disparity_with_BM(img_left, img_right, max_disparity=64, window_size=11):
    stereo = cv.StereoBM(
        minDisparity=0,
        numDisparities=max_disparity,
        blockSize=window_size,
        P1=3 * 4 * window_size ** 2,
        P2=3 * 32 * window_size ** 2,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv.StereoSGBM_MODE_HH)
    disp = stereo.compute(img_left, img_right).astype('float32') / 16
    return disp


def calculate_disparity(img_left, img_right, max_disparity=64, window_size=(11, 11), direction=DispDirection.left_to_right, criterium=DispCriterium.argmin, threads_num=1):
    params = (img_left, img_right, max_disparity, window_size, criterium)
    if direction == DispDirection.right_to_left:
        func = calculate_disparity_from_right_to_left
    elif direction == DispDirection.left_to_right:
        func = calculate_disparity_from_left_to_right
    else:
        raise Exception("Wrong direction argument!")

    if threads_num == 1:
        return func(*params)
    else:
        return calculate_disparity_multithread(func, params, window_size, img_left.shape, threads_num)


def calculate_disparity_multithread_worker(func, params, range, outputDict, first):
    outputDict[range] = func(*params, yRange=range, print_progress=first)
    print("Finished disparity range: ", range)


def calculate_disparity_multithread(func, params, window_size, dim, threads_num):
    halfS = math.floor(window_size[0]/2)
    yRange = (halfS, dim[0]-halfS)
    yRanges = []
    pools = []
    sizePerThread = math.ceil((yRange[1] - yRange[0]) / threads_num)
    manager = mlp.Manager()
    outputDict = manager.dict()
    first = True

    for i in range(yRange[0], yRange[1], sizePerThread):
        r = (i, min(i+sizePerThread, yRange[1]))
        yRanges.append(r)
        cur_params = (np.copy(params[0]), np.copy(params[1]), *params[2:])
        p = mlp.Process(target=calculate_disparity_multithread_worker, args=(
            func, cur_params, r, outputDict, first))
        p.start()
        pools.append(p)
        first = False
    for p in pools:
        p.join()

    output = np.zeros((dim[0], dim[1]))
    for r in yRanges:
        output[r[0]:r[1]] = outputDict[r]
    return output


def calculate_disparity_from_right_to_left(img_left, img_right, max_disparity, window_size, criterium, yRange=(-1, -1), print_progress=True):
    height = np.shape(img_left)[0]
    width = np.shape(img_left)[1]
    window_height = window_size[0]
    window_width = window_size[1]
    half_window_height = window_height // 2
    half_window_width = window_width // 2
    y_range = half_window_height, height - half_window_height
    moved_y = True

    if yRange[0] != -1 and yRange[1] != -1:
        y_range = yRange
        disparity = np.zeros((y_range[1]-y_range[0], width))
    else:
        disparity = np.zeros((height, width))
        moved_y = False
    iterator = range(y_range[0], y_range[1])
    if print_progress:
        iterator = tqdm(iterator)

    for y in iterator:
        for x in range(width - half_window_width, half_window_width, -1):
            template = img_left[y - half_window_height: y + half_window_height,
                                x - half_window_width: x + half_window_width]
            n_disparity = min(max_disparity, x - half_window_width)
            score = np.zeros(n_disparity)

            for offset in range(n_disparity, 0, -1):
                roi = img_right[y - half_window_height: y + half_window_height,
                                x - half_window_width - offset: x + half_window_width - offset]
                score[offset - 1] = abs_m(roi, template)

            if criterium == DispCriterium.argmax:
                disp = score.argmax()
            elif criterium == DispCriterium.argmin:
                disp = score.argmin()
            else:
                raise Exception("Wronf DispCriterium!")

            if moved_y:
                disparity[y-y_range[0], x] = disp
            else:
                disparity[y, x] = disp
    return disparity


def calculate_disparity_from_left_to_right(img_left, img_right, max_disparity, window_size, criterium, yRange=(-1, -1), print_progress=True):
    height = np.shape(img_left)[0]
    width = np.shape(img_left)[1]
    window_height = window_size[0]
    window_width = window_size[1]
    half_window_height = window_height // 2
    half_window_width = window_width // 2
    disparity = np.zeros((height, width))
    y_range = half_window_height, height - half_window_height
    moved_y = True

    if yRange[0] != -1 and yRange[1] != -1:
        y_range = yRange
        disparity = np.zeros((y_range[1]-y_range[0], width))
    else:
        disparity = np.zeros((height, width))
        moved_y = False
    if yRange != (-1, -1):
        y_range = yRange
    iterator = range(y_range[0], y_range[1])
    if print_progress:
        iterator = tqdm(iterator)

    for y in iterator:
        for x in range(half_window_width, width - half_window_width):
            template = img_right[y - half_window_height: y + half_window_height,
                                 x - half_window_width: x + half_window_width]
            n_disparity = min(max_disparity, width - x - half_window_width)
            score = np.zeros(n_disparity)

            for offset in range(n_disparity):
                roi = img_left[y - half_window_height: y + half_window_height,
                               x - half_window_width + offset: x + half_window_width + offset]
                score[offset - 1] = abs_m(template, roi)

            if criterium == DispCriterium.argmax:
                disp = score.argmax()
            elif criterium == DispCriterium.argmin:
                disp = score.argmin()
            else:
                raise Exception("Wronf DispCriterium!")

            if moved_y:
                disparity[y-y_range[0], x] = disp
            else:
                disparity[y, x] = disp
    return disparity


# Sum of square difference
def ssd(img_left, img_right):
    return np.sum((img_left - img_right) ** 2)


def abs_m(img_left, img_right):
    '''Should be faster than ssd and give the same result'''
    return np.sum(np.abs(img_left - img_right))


def calculate_disparity_with_depth(depth, f, baseline, doffs):
    disp = np.zeros(shape=depth.shape)
    for i in range(len(depth)):
        for j in range(len(depth[i])):
            if (depth[i][j] == 0):
                pass
            else:
                disp[i][j] = clamp(0, 255, (baseline * f) /
                                   (depth[i][j] - doffs))
                #disp[i][j] = (baseline * f) / (depth[i][j] - doffs)
    return disp


# >>> Depth methods <<<
def calculate_depth_with_disparity(disp, f, baseline, doffs):
    depth = np.zeros(shape=disp.shape)
    for i in range(len(disp)):
        for j in range(len(disp[i])):
            if (disp[i][j] == 0):
                depth[i][j] = (f * baseline) / 1
            else:
                depth[i][j] = (f * baseline) / (disp[i][j] + doffs)

    return depth


def calculate_depth_with_rgb24(rgb24, max_distance):
    array = image_to_bgra(rgb24)
    array = array.astype(np.float32)

    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth * max_distance


def calculate_rgb24_with_depth(depth, max_distance):
    array = depth.astype(np.float32)
    array /= max_distance
    array *= 16777215.0

    rgb24 = np.zeros(shape=[array.shape[0], array.shape[1], 3])
    for i in range(len(array)):
        for j in range(len(array[i])):
            data = array[i][j]
            r = data % 256
            data = (data - r) / 256
            g = data % 256
            b = (data - g) / 256

            rgb24[i][j][0] = clamp(0, 255, r)
            rgb24[i][j][1] = clamp(0, 255, g)
            rgb24[i][j][2] = clamp(0, 255, b)

    return rgb24


# >>> Point cloud methods <<<
def save_depth_to_ply(depth, fov, out_path):
    pcd = []
    height, width = depth.shape
    cx, cy = calculate_cx_cy_with_image_size(width, height)
    for i in range(height):
        for j in range(width):
            z = depth[i][j]
            x = (j - cx) * z / fov  # fx
            y = (i - cy) * z / fov  # fy
            pcd.append([x, y, z])
    pcd_o3d = o3d.geometry.PointCloud()                 # Create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(
        pcd)    # Set pcd_np as the point cloud points
    o3d.io.write_point_cloud(out_path, pcd_o3d)
    o3d.visualization.draw_geometries([pcd_o3d])        # Visualize


def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def save_to_ply(disp, img_left, f, out_path):
    print('Generating 3d point cloud')
    h, w = img_left.shape[:2]
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],    # Turn points 180 deg around x-axis,
                    [0, 0, 0, -f],          # so that y-axis looks up
                    [0, 0, 1, 0]])
    points = cv.reprojectImageTo3D(disp, Q)
    colors = cv.cvtColor(img_left, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = out_path

    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)
    visualize_ply(out_path)


def visualize_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])
