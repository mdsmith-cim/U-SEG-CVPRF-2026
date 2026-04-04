# Datasets

The datasets used follow the Mask2Former format. They are not distributed due to licensing, and so will need to be downloaded. These scripts are used to process them into a compatible format.

## Mask2Former datasets
Mask2Former is already compatible with several datasets, such as COCO. Please see the [relevant documentation](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md) for instructions on how to prepare them.

## Custom datasets

In our work, we use two datasets: VIPER and Cityscapes. The former we add, the latter is already compatible with Mask2Former. However, in our work we inivestigate time series data, which requires a custom implementation as each frame must be associated with some number of previous frames.

## Scripts of interest
- `create_VIPER_dataset.py`: once downloaded from the [website](https://playing-for-benchmarks.org), this script will convert it.
- `create_debug_coco*.py`: used to create very small versions of COCO for debugging without running through the entire dataset. Useful if errors occur only with specific images.
- `generate_optical_flow_cityscapes.py`: Uses RAFT to generate optical flow for Cityscapes to use as a mapping between frames. An equivalent is not needed for VIPER as ground truth data is provided.
- `generate_VIPER_cityscapes_distributionshift.py`: We examine distribution shifts between VIPER and Cityscapes, where we run inference on one dataset with a model trained on the other. As the class mappings are different, we create shifted versions of each so that the already trained model can be used and the classes line up as best as possible. This script creates shifted versions of both datasets.
- `delete_future_cityscapes_seq_images.py`: The Cityscapes dataset provides a number of unlabeled frames that occur temporally before and after the labeled frames. This script deletes all of those that occur in the future as well as those from the training and test sets as we do not use them to save disk space.