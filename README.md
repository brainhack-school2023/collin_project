# Project for Brainhack-School 2023
## Myelin segmentation on histology data using a foundation model

### About me
| <a href="https://github.com/hermancollin"><img src="https://scontent.fyhu1-1.fna.fbcdn.net/v/t1.18169-9/26907692_328382414314217_4507159261681077282_n.jpg?_nc_cat=105&ccb=1-7&_nc_sid=09cbfe&_nc_ohc=1JSb6jfqGx8AX90FK1z&_nc_ht=scontent.fyhu1-1.fna&oh=00_AfD7UBU5zGSbQIIrYIK6uLOM3IRqaReHp8fKRQHUYmH3EQ&oe=64A82817" width="400px;" alt=""/><br/><sub>Armand Collin</sub></a> | Hi! My name is Armand and I am a Master student in biomedical engineering at NeuroPoly (Polytechnique Montréal). I come from an electrical engineering background and my interests include Deep Learning for computer vision and software development. I work on [axondeepseg](https://github.com/axondeepseg/axondeepseg) with neurohistology data. |
|-----------|:---------------|


## Project Summary

### Introduction
Histology (microscopy) data is widely used by neuropathologists to study demylienation in the nervous system.
My project aims to leverage a general-purpose foundation model to segment myelin on histology images. Foundation models are large DL models trained on large-scale data. They learn a general representation that can be adapted to a variety of downstream tasks. OpenAI's GPT serie, for example, are examples of foundation models for NLP.

### Main Objectives
- Prepare a microscopy dataset to a format compatible with the Segment-Anything-Model
- Fine-tune the foundation model as a proof-of-concept

### Tools
- `git`/`git-annex` for version control and data retrieval
- BIDS standard
- [axondeepseg](https://github.com/axondeepseg/axondeepseg) for data preprocessing
- [SAM checkpoint](https://github.com/facebookresearch/segment-anything/tree/main) to fine-tune
- Jupyter Notebook for prototyping
- Main pythong packages: `torch`, `numpy`, `pandas`, `PIL`, `cv2`, `monai`

### Data
The data used for this project is the `data_axondeepseg_tem` dataset privately hosted on an internal server with git-annex. It was used to train [this model](https://github.com/axondeepseg/default-TEM-model). It's also our biggest annotated dataset for myelin segmentation (20 subjects, 1360 MPx of manually segmented images).

### Project deliverables
1. `axondeepseg` PR[742](https://github.com/axondeepseg/axondeepseg/pull/742) adds a feature to save raw instance maps.
With this feature, we can take a semantic segmentation and turn it into a raw 16bit PNG format where all axons are individually labelled. Below, we can see an example of an input semantic segmentation and its associated colorized instance segmentation. This allows us to subdivide the segmentation mask into its individual components.

| Semantic seg | Instance seg |
|:-:|:-:|
| <img src="https://github.com/brainhack-school2023/collin_project/assets/83031821/d09274af-b062-43c3-815f-a45850e5ef3a"> | <img src="https://github.com/brainhack-school2023/collin_project/assets/83031821/fc04f880-737a-43f4-a5b9-2a764c9f9434"  > |


2. Preprocessing script (located in `scripts/preprocessing.py`) which allows us to take the myelin segmentation and the instance map and extract the myelin map and the bbox/centroid information that we will feed to SAM as input prompts. Below, we can see a QC visualization of the output. This preprocessing script takes as input a BIDS dataset and outputs a BIDS-compatible derivatives folder.

![sub-nyuMouse07_sample-0007_qc](https://github.com/brainhack-school2023/collin_project/assets/83031821/7e5cf53f-b6f5-4cf6-bf9f-db33a9373edf)


## Results
[...]

## Conclusions
[...]

## How to reproduce
For a complete guide to reproduce these results, please see the README in the `scripts` folder.
