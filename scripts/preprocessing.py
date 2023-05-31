from AxonDeepSeg import ads_utils
from AxonDeepSeg.morphometrics.compute_morphometrics import get_axon_morphometrics

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

import json
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

import bids_utils

def extract_myelin_map_and_prompts(axon_path, myelin_path, px_size):
    '''
    Main preprocessing function. It returns the myelin map with all individual 
    myelin masks and their associated bbox/centroid to feed to SAM as prompts.
    '''
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

    mask = myelin_mask == 255
    myelin_map = instance_map.astype(np.uint16) * mask
    myelin_im = instance_seg_im * np.repeat(mask[:,:,np.newaxis], 3, axis=2)

    # collect bbox info
    bboxes = morphometrics.iloc[:, -4:]
    # collect axon centroids
    centroids = morphometrics.iloc[:, :2]

    # create reduced dataframe
    prompts_df = pd.concat([centroids.astype('int64'), bboxes], axis=1)
    prompts_df.rename(columns={'x0 (px)': 'x0', 'y0 (px)': 'y0'}, inplace=True)
    col_order = ['x0', 'y0', 'bbox_min_x', 'bbox_min_y', 'bbox_max_x', 'bbox_max_y']
    prompts_df = prompts_df.reindex(columns=col_order)

    return index_im, myelin_map, myelin_im, prompts_df

def save_bbox_img(myelin_img, prompts_df, index_im, fname):
    '''
    For QC purposes, this function saves the image with bboxes overlayed.
    :param myelin_img:  Path to myelin mask
    :param prompts_df:  Prompt dataframe containing bboxes/centroids
    :param index_im:    Index image (output of get_axon_morphometrics)
    :param fname:       Output filename
    '''
    mask = Image.fromarray(myelin_img)
    rgbimg = Image.new("RGBA", mask.size)
    rgbimg.paste(mask)

    N = prompts_df.shape[0]
    draw = ImageDraw.Draw(rgbimg)
    for i in range(N):
        # Create a rectangle around axon #i
        draw.rectangle(
            [
                prompts_df.iloc[i, 2], 
                prompts_df.iloc[i, 3],
                prompts_df.iloc[i, 4],
                prompts_df.iloc[i, 5],
            ],
            outline='red',
            width=2
        )
    index_im = Image.fromarray(index_im)
    rgbimg.paste(index_im, mask=index_im)
    rgbimg.save(fname)

def main(datapath, output_path=None):
    data_dict = bids_utils.index_bids_dataset(datapath)
    if output_path is None:
        output_path = Path().cwd()
    output_path = output_path / 'derivatives' / 'maps'
    output_path.mkdir(parents=True, exist_ok=True)
    
    for sub in tqdm(data_dict.keys()):
        subject_path = output_path / sub
        subject_path.mkdir(exist_ok=True)

        px_size = data_dict[sub]['px_size']
        samples = (s for s in data_dict[sub].keys() if 'sample' in s)
        for sample in samples:
            samples_path = subject_path / 'micr'
            samples_path.mkdir(exist_ok=True)

            ax_seg = data_dict[sub][sample]['axon']
            my_seg = data_dict[sub][sample]['myelin']

            (idx_im, 
             myelin_map, 
             myelin_im, 
             prompts) = extract_myelin_map_and_prompts(ax_seg, my_seg, px_size)

            qc_fname = samples_path / f'{sub}_{sample}_qc.png'
            map_fname = samples_path / f'{sub}_{sample}_myelinmap.png'
            prompts_fname = samples_path / f'{sub}_{sample}_prompts.csv'

            # save all derivatives
            save_bbox_img(myelin_im, prompts, idx_im, qc_fname)
            ads_utils.imwrite(map_fname, myelin_map.astype(np.uint16))
            prompts.to_csv(prompts_fname)
            nb_axons = prompts.shape[0]
            if nb_axons >= 256:
                print(f'WARNING: {sub}_{sample} has {nb_axons} axons. It will be saved in 16bit format.')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-d',
        dest='datapath',
        type=str,
        required=True,
        help='Path to the BIDS dataset'
    )
    parser.add_argument(
        '-o',
        dest='output_path',
        type=str,
        help='Where to save the output'
    )

    args = parser.parse_args()
    main(args.datapath, args.output_path)