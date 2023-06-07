# Project for Brainhack-School 2023
## Myelin segmentation on histology data using a foundation model

### About me
| <a href="https://github.com/hermancollin"><img src="https://scontent.fyhu1-1.fna.fbcdn.net/v/t1.18169-9/26907692_328382414314217_4507159261681077282_n.jpg?_nc_cat=105&ccb=1-7&_nc_sid=09cbfe&_nc_ohc=1JSb6jfqGx8AX90FK1z&_nc_ht=scontent.fyhu1-1.fna&oh=00_AfD7UBU5zGSbQIIrYIK6uLOM3IRqaReHp8fKRQHUYmH3EQ&oe=64A82817" width="400px;" alt=""/><br/><sub>Armand Collin</sub></a> | Hi! My name is Armand and I am a Master student in biomedical engineering at NeuroPoly (Polytechnique Montréal). I come from an electrical engineering background and my interests include Deep Learning for computer vision and software development. I work on [axondeepseg](https://github.com/axondeepseg/axondeepseg) with neurohistology data. |
|-----------|:---------------|


## Project Summary

### Introduction
Histology (microscopy) data is widely used by neuropathologists to study demylienation in the nervous system.
My project aims to leverage a general-purpose foundation model to segment myelin on histology images. Foundation models are large DL models trained on large-scale data. They learn a general representation that can be adapted to a variety of downstream tasks. OpenAI's GPT serie, for example, are examples of foundation models for natural language processing.

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
- GPU cluster for training (1X RTX A6000)

### Data
The data used for this project is the `data_axondeepseg_tem` dataset privately hosted on an internal server with git-annex. It was used to train [this model](https://github.com/axondeepseg/default-TEM-model). It's also our biggest annotated dataset for myelin segmentation (20 subjects, 1360 MPx of manually segmented images). An older version of this dataset is publicly available on this [OSF repository](https://osf.io/bj9eu/), under the `data/raw/` directory. For more information on how to acces the data, see the *How to reproduce* section below.

### Project deliverables
1. `axondeepseg` PR [#742](https://github.com/axondeepseg/axondeepseg/pull/742) adds a feature to save raw instance maps.
With this feature, we can take a semantic segmentation and turn it into a raw 16bit PNG format where all axons are individually labelled. Below, we can see an example of an input semantic segmentation and its associated colorized instance segmentation. This allows us to subdivide the segmentation mask into its individual components. This PR will eventually be merged to the master branch.

| Semantic seg | Instance seg |
|:-:|:-:|
| <img src="https://github.com/brainhack-school2023/collin_project/assets/83031821/d09274af-b062-43c3-815f-a45850e5ef3a"> | <img src="https://github.com/brainhack-school2023/collin_project/assets/83031821/fc04f880-737a-43f4-a5b9-2a764c9f9434"  > |


2. Preprocessing script (located in `scripts/preprocessing.py`) which allows us to take the myelin segmentation and the instance map and extract the myelin map and the bbox/centroid information that we will feed to SAM as input prompts. Below, we can see a QC visualization of the output. This preprocessing script takes as input a BIDS dataset and outputs a BIDS-compatible derivatives folder.

| Bounding boxes |
|:-:|
| <img src="https://github.com/brainhack-school2023/collin_project/assets/83031821/7e5cf53f-b6f5-4cf6-bf9f-db33a9373edf"  width="60%"> |

3. Image embedding pre-computation script (located in `scrips/precompute_embeddings.py`) which will pre-compute the image embeddings. The reason we do this is that, as we can see below, the most intensive part of the forward pass is through the image encoder, a big vision transformer. Since we do not want to fine-tune this part of the model (only the mask decoder), we can greatly reduce training time by pre-computing the image embeddings and using them directly during the fine-tuning. This means that we never load the images during training.

<div align="center">
  
| SAM architecture |
|:-:|
| <img src="https://learnopencv.com/wp-content/uploads/2023/04/segment-anything-pipeline.gif"> |

</div>

4. Training script (located in `scripts/training_gpu.py`). The model was fine-tuned for 40 epochs with the AdamW optimizer using the Dice loss. Gradient accumulation was also used because properly batching the input was not practical.
5. A myelin segmentation! See section below for results.


## Results

### Fine-tuning dataset
To download the source dataset, see the **How to reproduce** section below. The prompts (bounding boxes and centroids) and pre-computed image embeddings are available as release assets of this project. Technically, the source dataset with the raw images and labels is not needed and the fine-tuning can be done using only the derivatives included here.

### Segmentation results and model checkpoint
Below, we can see a comparison between the output of SAM before and after the fine-tuning. This segmentation was performed on an image held out from the training set. As we can see, this method works very well. Most of the myelin objects are perfectly segmented. Note that these images were obtained using the notebook located in `notebook/training/compare_against_baseline.ipynb`.

| Before fine-tuning | After fine-tuning |
|:-:|:-:|
| <img src="https://github.com/brainhack-school2023/collin_project/blob/main/results/before_finetuning.png?raw=true"> | <img src="https://github.com/brainhack-school2023/collin_project/blob/main/results/after_finetuning.png?raw=true"  > |

<div align="center">

| <img src="https://github.com/brainhack-school2023/collin_project/blob/main/results/losses_with_diceloss.png?raw=true"> |
|:-:|
  
</div>

The final checkpoint of the fine-tuned model is also available as a release asset of this project (see `sam_vit_b_01ec64_finetuned_diceloss.pth`).

## Conclusions
### Can we leverage a foundation model for segmentation of histology images
The answer is defintely yes. The results obtained with this project are very promising and could lead to a major improvement compared to our current models. The most impressive part about this method is that almost no hyperparameter tuning was involved: the training converged easily out-of-the-box. Obviously, optimizing hyperparameters will produce a better model, but the time constraint on this project did not allow to investigate this further. The reason why this is impressive is that usually, training a UNet from scratch and optimizing its performance is an involved process and requires Deep Learning expertise, whereas here, SAM is powerful enough to alleviate this challeging procedure.

### What I learned
- Familiarity with SAM architecture and codebase
- Managing and planning a reproducible project from scratch
- MRI basics from the Week 2 modules

### Future steps
- I would be curious to see how far we can push the model performance by tuning some hyperparameters
- Fine-tuning SAM on our other histology datasets
- Comparing this method against the segmentation models currently used by `axondeepseg`
- Try to fine-tune a general-purpose myelin segmentation model that works across modalities/resolutions
- If the segmentation results are better or very competitive to our current baseline, experiment with object detection methods to automatically generate the bounding boxes
- Eventually add this/these models to `axondeepseg` to replace our older models and to make the method available to any user

## How to reproduce
For a complete guide to reproduce these results, please see the README in the `scripts` folder.
