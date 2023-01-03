import numpy as np
from tqdm import tqdm


def calculate_disparity_bm_from_right_to_left(left_img, right_img, max_disparity=64, window_size=(11, 11)):
    assert np.shape(left_img)[0] == np.shape(right_img)[0] and np.shape(left_img)[1] == np.shape(right_img)[1]
    window_height = window_size[0]
    window_width = window_size[1]
    height = np.shape(left_img)[0]
    width = np.shape(left_img)[1]
    half_window_height = window_height // 2
    half_window_width = window_width // 2
    disparity = np.zeros((height, width))

    for y in tqdm(range(half_window_height, height - half_window_height)):
        for x in range(width - half_window_width, half_window_width, -1):
            template = left_img[y - half_window_height:y + half_window_height,
                                x - half_window_width:x + half_window_width]
            n_disparity = min(max_disparity, x - half_window_width)
            score = np.zeros(n_disparity)
            for offset in range(n_disparity, 0, -1):
                roi = right_img[y - half_window_height:y + half_window_height, x - half_window_width - offset:x + half_window_width-offset]
                score[offset-1] = ssd(roi, template)
            disparity[y, x] = score.argmin() #argmin albo argmax - zależy
    return disparity


def ssd(img1, img2):
    return np.sum((img1 - img2) ** 2)


def disp_to_depth(disp, baseline, f, doffs):
    return baseline * f / (disp + doffs)


def depth_to_disp(depth, baseline, f, doffs):
    return baseline * f / depth - doffs