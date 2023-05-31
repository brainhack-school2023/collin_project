import torch
import matplotlib.pyplot as plt

from segment_anything import SamPredictor, sam_model_registry
from preprocessing import index_bids_dataset

def main(datapath, outpath=None):
    data_dict = index_bids_dataset(datapath)
    if outpath is None:
        outpath = Path().cwd()
    outpath = outpath / 'derivatives' / 'embeddings'
    outpath.mkdir(parents=True, exist_ok=True)    
    
    for sub in tqdm(data_dict.keys()):
        subject_path = output_path / sub
        subject_path.mkdir(exist_ok=True)
        px_size = data_dict[sub]['px_size']
        samples = (s for s in data_dict[sub].keys() if 'sample' in s)
        for sample in samples:
            samples_path = subject_path / 'micr'
            samples_path.mkdir(exist_ok=True)



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