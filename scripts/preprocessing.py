from AxonDeepSeg import ads_utils
from AxonDeepSeg.morphometrics.compute_morphometrics import get_axon_morphometrics

from PIL import Image, ImageDraw
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
        return_index_image=True,
        return_border_info=True,
        return_instance_seg=True
    )
    morphometrics, index_im, instance_seg_im, instance_map = morph_output

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

    return index_im, myelin_map, prompts_df

def save_bbox_img(myelin_img, prompts_df, index_im, fname):
    '''
    For QC purposes, this function saves the image with bboxes overlayed.
    :param myelin_img:  Path to myelin mask
    :param prompts_df:  Prompt dataframe containing bboxes/centroids
    :param index_im:    Index image (output of get_axon_morphometrics)
    :param fname:       Output filename
    '''
    mask = Image.open(myelin_img)
    rgbimg = Image.new("RGBA", mask.size)
    rgbimg.paste(mask)

    N = prompts_df.shape[0]
    draw = ImageDraw.Draw(rgbimg)
    for i in range(N):
        # Create a rectangle around axon #i
        draw.rectangle(
            [
                prompts_df.iloc[i, 3], 
                prompts_df.iloc[i, 2],
                prompts_df.iloc[i, 3] + prompts_df.iloc[i, -2],
                prompts_df.iloc[i, 2] + prompts_df.iloc[i, -1],
            ],
            outline='red'
        )
    index_im = Image.fromarray(index_im)
    rgbimg.paste(index_im, mask=index_im)
    rgbimg.save(fname)

