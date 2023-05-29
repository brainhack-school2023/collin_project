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
    morphometrics, instance_seg_im, instance_map = morph_output

    myelin_map = instance_map * (myelin_mask == 255)

    # collect bbox info
    bboxes = morphometrics.iloc[:, -4:]
    min_x, min_y = bboxes.iloc[:, -3], bboxes.iloc[:, -4]
    w = bboxes.iloc[:, -1] - bboxes.iloc[:, -3]
    h = bboxes.iloc[:, -2] - bboxes.iloc[:, -4]
    # collect axon centroids
    centroids = morphometrics.iloc[:, :2]

    # create reduced dataframe
    width_heigth = pd.DataFrame({'width': w, 'heigth': h})
    prompts_df = pd.concat([centroids.astype('int64'), bboxes, width_heigth], axis=1)
    prompts_df.rename(
        columns={
            'bbox_min_y': 'min_y', 
            'bbox_min_x': 'min_x',
            'x0 (px)': 'x0',
            'y0 (px)': 'y0',
        }, 
        inplace=True
    )
    prompts_df.drop(columns=['bbox_max_y', 'bbox_max_x'], inplace=True)

    return myelin_map, prompts_df