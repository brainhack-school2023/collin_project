from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import torch

from segment_anything import SamPredictor, sam_model_registry
from bids_utils import index_bids_dataset

# Attach these 2 functions to the SamPredictor class to load/save embeddings
def save_image_embedding(self, path):
    if not self.is_image_set:
        raise RuntimeError("An image must be set with .set_image(...) before embedding saving.")
    res = {
        'original_size': self.original_size,
        'input_size': self.input_size,
        'features': self.features,
        'is_image_set': True,
    }
    torch.save(res, path)

def load_image_embedding(self, path):
    res = torch.load(path, self.device)
    for k, v in res.items():
        setattr(self, k, v)

SamPredictor.save_image_embedding = save_image_embedding
SamPredictor.load_image_embedding = load_image_embedding


def main(datapath, outpath=None):

    print('Loading the SAM model.')
    model_type = 'vit_b'
    checkpoint = 'sam_vit_b_01ec64.pth'
    device = 'cpu'
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)
    predictor = SamPredictor(sam_model)

    # index BIDS dataset and compute embeddings
    data_dict = index_bids_dataset(datapath)
    if outpath is None:
        outpath = Path().cwd()
    outpath = outpath / 'derivatives' / 'embeddings'
    outpath.mkdir(parents=True, exist_ok=True)    
    
    print('Computing image embeddings.')
    for sub in tqdm(data_dict.keys()):
        subject_path = outpath / sub
        subject_path.mkdir(exist_ok=True)
        px_size = data_dict[sub]['px_size']
        samples = (s for s in data_dict[sub].keys() if 'sample' in s)
        for sample in samples:
            samples_path = subject_path / 'micr'
            samples_path.mkdir(exist_ok=True)

            image = cv2.imread(data_dict[sub][sample]['image'])
            # this will pass the image through the vision encoder
            predictor.set_image(image)
            emb_fname = samples_path / f'{sub}_{sample}_TEM_embedding.pt'
            predictor.save_image_embedding(emb_fname)


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