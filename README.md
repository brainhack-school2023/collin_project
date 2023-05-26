# Project for Brainhack-School 2023


<a href="https://github.com/hermancollin">
   <img src="https://avatars.githubusercontent.com/u/83031821?v=4" width="100px;" alt=""/>
   <br /><sub><b>Armand Collin</b></sub>
</a>

Hi! My name is Armand and I am a Master student at NeuroPoly (Polytechnique Montréal).

-----

# Saving instance maps
`axondeepseg` PR #742 adds this feature and is located here: https://github.com/axondeepseg/axondeepseg/pull/742

With this feature, we can take a semantic segmentation and turn it into a raw 16bit PNG format where all axons are individually labelled. Below, we can see an example of an input semantic segmentation and its associated colorized instance segmentation. This allows us to subdivide the segmentation mask into its individual components.

| Semantic seg | Instance seg |
|:-:|:-:|
| <img src="https://github.com/brainhack-school2023/collin_project/assets/83031821/d09274af-b062-43c3-815f-a45850e5ef3a"> | <img src="https://github.com/brainhack-school2023/collin_project/assets/83031821/fc04f880-737a-43f4-a5b9-2a764c9f9434"  > |

# Convert labels from a dataset
The data used for this project is the `data_axondeepseg_tem` dataset privately hosted on an internal server with git-annex. It was used to train [this model](https://github.com/axondeepseg/default-TEM-model). It's also our biggest annotated dataset for myelin segmentation (20 subjects, 1360 MPx of manually segmented images).
