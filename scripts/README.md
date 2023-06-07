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

## Pre-computing the image embeddings

## Training the model
