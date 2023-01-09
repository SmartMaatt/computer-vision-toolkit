import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

import IO
import api

# >>> Przykładowe zadanie i proces wykonania na podstawie zestawu Artroom2 <<<
# Korzystając z metody StereoSGBM udostępnionej przez bibliotekę OpenCV wyznacz mapę rozbieżności oraz mapę głębi.
# Dobierz odpowiednie parametry metody StereoSGBM.
# Porównaj otrzymane wyniki z referencyjnymi wartościami rozbieżności zapisanymi w formacie PFM 

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
x = 830
y = 530


# Obliczanie dysparycji z danych przy pomocy SGBM
disp = api.calculate_disparity_with_SGBM(img_left, img_right, max_disparity=ndisp, window_size=2)
for i in range(len(disp)):
    for j in range(len(disp[i])):
        disp[i][j] = api.clamp(0, 255, disp[i][j])

# Odczytanie dysparycji do porównania
ref_disp = cv.imread(r"./Artroom2/disp0.pfm", cv.IMREAD_UNCHANGED)
ref_disp = np.asarray(ref_disp)
ref_disp = ref_disp / 256

matplotlib.pyplot.imshow(disp)
plt.show()
cv.imwrite("Artroom2/Results/Zad 1/disparity.png", disp)
IO.save_disp_to_json(disp, "Artroom2/Results/Zad 1/disparity_data.json")
print(f"Zad 1 - Wartość dysparycji wyznaczonej przez StereoSGBM dla [{x},{y}]: {disp[y][x]}")

matplotlib.pyplot.imshow(ref_disp)
plt.show()
cv.imwrite("Artroom2/Results/Zad 1/ref_disparity.png", ref_disp)
IO.save_disp_to_json(ref_disp, "Artroom2/Results/Zad 1/ref_disparity_data.json")
print(f"Zad 1 - Wartość dysparycji referencyjnej dla [{x},{y}]: {ref_disp[y][x]}")


# Obliczanie głębi na podstawie uzyskanych dysparycji
depth = api.calculate_depth_with_disparity(disp, f, baseline, doffs)
ref_depth = api.calculate_depth_with_disparity(ref_disp, f, baseline, doffs)

matplotlib.pyplot.imshow(depth)
plt.show()
cv.imwrite("Artroom2/Results/Zad 1/depth.png", depth / 255)
print(f"Zad 2 - Wartość depth wyznaczonej przez StereoSGBM dla [{x},{y}]: {depth[y][x]}")

matplotlib.pyplot.imshow(ref_depth)
plt.show()
cv.imwrite("Artroom2/Results/Zad 1/ref_depth.png", ref_depth)
print(f"Zad 2 - Wartość depth referencyjnej dla [{x},{y}]: {ref_depth[y][x]}")