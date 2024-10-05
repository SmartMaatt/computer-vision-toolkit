import matplotlib
from matplotlib import pyplot as plt
import cv2 as cv

import IO
import toolkit

# >>> Przykładowe zadanie i proces wykonania na podstawie zestawu Car 2 <<<
# Odczytaj wartości z mapy głębi z pliku PNG.
# Wyznacz mapę rozbieżności (disparity map) na podstawie mapy głębi (depth map) odczytanej z pliku PNG. 

# Zad 1
bit24 = IO.read_image_to_np_array("Car2/depth.png")
min_distance = 0
max_distance = 1000
x = 500
y = 500                                                                             # Do wyznaczania dowolnego pkt najlepiej użyć painta i wybrać kursorem interesujące współrzędne ;)

depth = toolkit.calculate_depth_with_rgb24(bit24, max_distance)                         # Obliczanie depth na podstawie depth_24bit

matplotlib.pyplot.imshow(depth)
plt.show()
cv.imwrite("Car2/Results/depth.png", depth)
print(f'Zad 1 - Odległość do wybranego pkt [{x},{y}] {depth[y][x]}')                # Pamiętać współrzędne kodowane odwrotnie!


# Zad 2
baseline = 0.6
fov = 120
f = toolkit.calculate_focalX_with_FovW_deg(depth.shape[0], fov)                         # Dane z .txt lub obliczone

disp = toolkit.calculate_disparity_with_depth(depth, f, baseline, 0)                    # Obliczanie disparity na podstawie depth
IO.save_disp_to_json(disp, "Car2/Results/disparity_data.json")

matplotlib.pyplot.imshow(disp)
plt.show()
cv.imwrite("Car2/Results/disparity.png", disp)
print(f'Zad 2 - Wartość dysparycji dla [{x},{y}] {disp[y][x]}')