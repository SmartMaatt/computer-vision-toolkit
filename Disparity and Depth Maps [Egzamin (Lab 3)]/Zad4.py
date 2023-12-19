import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

import IO
import api

# >>> Przykładowe zadanie i proces wykonania na podstawie zestawu Artroom2 <<<
# Wygeneruj chmurę punktów dla zestawu artroom2 a następnie zapisz ją do pliku ply.

IMG_LEFT = r"./Artroom2/im0.png"
IMG_RIGHT = r"./Artroom2/im1.png"
img_left = cv.imread(IMG_LEFT)
img_right = cv.imread(IMG_RIGHT)

doffs = 0
baseline = 529.50
f = 1734.04         # Focal z macierzy
width = 1920
height = 1080
ndisp = 190

# Obliczanie dysparycji z danych przy pomocy SGBM
disp = api.calculate_disparity_with_SGBM(img_left, img_right, max_disparity=ndisp, window_size=2)
for i in range(len(disp)):
    for j in range(len(disp[i])):
        disp[i][j] = api.clamp(0, 255, disp[i][j])

# Wygeneruj chmurę punktów
api.save_to_ply(disp, img_left, f, "Artroom2/Results/Zad 3/disp_cloud.ply")     # Czasami trzeba trochę przybliżyć ;)

# Ewentualnie zapis chmury punktów z głębi
depth = api.calculate_depth_with_disparity(disp, f, baseline, doffs)

# Wygeneruj chmurę punktów
fov = api.calculate_fovW_with_focalX(width, f)
api.save_depth_to_ply(depth, fov, "Artroom2/Results/Zad 3/depth_cloud.ply")