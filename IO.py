import json
import cv2 as cv
import numpy as np
from json import JSONEncoder


def save_disp_to_json(disp, path):
    # >>> Saving result to json <<<
    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    json_data = {
        'disp': disp
    }

    # Writing to json
    with open(path, "w") as outfile:
        json.dump(json_data, outfile, indent=4, cls=NumpyArrayEncoder)


def read_disp_data(path):
    # Read JSON file
    print(f'Reading calibration file: {path}')
    with open(fr'{path}', 'r') as f:
      calibration_data = json.load(f)

    disp = np.array(calibration_data['disp'])
    return disp


def read_image_to_np_array(path):
    return np.asarray(cv.imread(path, cv.IMREAD_UNCHANGED))