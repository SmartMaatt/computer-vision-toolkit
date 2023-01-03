import disparity
#import pyplot from matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

#https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/
CONES1_IMG = r"./Cones/im2.png"
CONES2_IMG = r"./Cones/im6.png"

#https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Motorcycle-perfect/
BIKE1_IMG = r"./Motocycle/im0.png"
BIKE2_IMG = r"./Motocycle/im1.png"

# Dane w calib.txt datasetu - w tym przypadku dla bike1 i bike2
DOFFS = 124.343
BASELINE = 193.001
F = 3979.911

if __name__ == '__main__':
    img1 = cv.imread(CONES1_IMG)
    img2 = cv.imread(CONES2_IMG)
    disp = disparity.calculate_disparity_bm_from_right_to_left(img1, img2)
    # depth tylko jeżeli jest baseline, fokal i doffs - dla bike byly w pliku calib.txt https://vision.middlebury.edu/stereo/data/scenes2014/
    # depth = disparity.disp_to_depth(disp, BASELINE, F, DOFFS)
    color_map = plt.cm.get_cmap('turbo', 8)
    plt.imshow(disp, cmap=color_map)
    plt.show()
    #jak zapisac disparity i depth do pliku? - na zajeciach mowil o 3 mozliwosciach:
    # #bilbioteką (?)tif(?) zapisać tablicę
    #zapisać jako obrraz (.png)
    #znormalizować do 8 bitów i zapisać jako obraz