from AxonDeepSeg import ads_utils
from AxonDeepSeg.morphometrics.compute_morphometrics import get_axon_morphometrics

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

def extract_myelin_map_and_prompts(axon_path, myelin_path, px_size):
    axon_mask = ads_utils.imread(axon_path)
    myelin_mask = ads_utils.imread(myelin_path)
    morph_output = get_axon_morphometrics(
        axon_mask, 
        im_myelin=myelin_mask, 
        pixel_size=px_size,
        return_border_info=True,
        return_instance_seg=True
    )
    df, instance_seg_im, instance_map = morph_output

    myelin_map = instance_map * (myelin_mask == 255)

    bboxes = morphometrics.iloc[:, -4:]
    min_x, min_y = bboxes.iloc[:, -3], bboxes.iloc[:, -4]
    w = bboxes.iloc[:, -1] - bboxes.iloc[:, -3]
    h = bboxes.iloc[:, -2] - bboxes.iloc[:, -4]