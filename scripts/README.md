# How to reproduce
This guide will go through all the steps required to reproduce all the results presented in this repository. 
Please note that the [notebook](https://github.com/brainhack-school2023/collin_project/tree/8127ee663a190492ff8954503a65e36b4e198e9f/notebook) folder in this repository contains jupyter notebooks that explain step-by-step how the pipeline works.

## Pre-processing the data
If you have access to the data server, this is how to obtain the dataset used in this project:
```
git clone git@data.neuro.polymtl.ca:datasets/data_axondeepseg_tem
cd data_axondeepseg_tem
git annex get .
```
Also, please note that an older version of this dataset is publicly available on this OSF repository: https://osf.io/bj9eu/

For this step, you will need a virtual environment with `axondeepseg` installed. Follow the [ADS installation insructions](https://axondeepseg.readthedocs.io/en/latest/documentation.html#installation), then activate the virtual environment with
```
conda activate ads_venv
```
Finally, to obtain the prompts and preprocess the myelin segmentation, run the preprocessing script and specify the path to the dataset:
```
python preprocessing.py -d path/to/dataset
```
This will output a BIDS derivative folder.

## Pre-computing the image embeddings
For the following 2 steps, we will work in another virtual environment. We will need the `segment-anything` dependency. Create a virtual environment and install SAM and its dependencies:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

You will also need a model checkpoint. For this project, we worked with the smallest available model (ViT-B).
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
Finally, run the embedding computation script by providing the path to the dataset:
```
python precompute_embeddings.py -d path/to/dataset
```
This will output a BIDS derivative folder.

## Training the model
In the same virtual environment as the one used in the previous step, install the following dependency (used for the loss function)
```
pip install monai
```
Then, you can run the training script:
```
python training_gpu.py
```
Note that you may need to change the GPU device and some harcoded paths in the `training_gpu.py` script. Training for 40 epochs on a RTX A6000 (2020), this took around 8 hours. This script will output model checkpoints every 5 epochs and predictions at every epoch to monitor progress.
