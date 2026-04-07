# U-SEG: Uncertainty in SEGmentation - A systematic multi-variable exploration

This repository contains the code for the U-SEG paper (CVPR Findings 2026) by Michael Smith and Frank Ferrie. It is based on the [Mask2Former project from Facebook](https://github.com/facebookresearch/mask2former). Thus, there is a lot of inherited code not used in this project.


## Installation

See [installation instructions](INSTALL.md).

## Models

- Where possible, we used models provided in the [original Model Zoo from Facebook](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md). 
- In some cases, when training a model we swapped out the backbone but kept the Mask2Former head for transfer learning. Those models are suffixed with `backbone_stripped` e.g `model_final_54b88a_backbone_stripped.pkl`. In those cases, the original model (e.g. `model_final_54b88a.pkl`) can be found from the Facebook Model Zoo and the `strip-backbone-weights.py` script in `tools/` can be used to generate the weights file we used. 
- All models trained by us are available on [HuggingFace](https://huggingface.co/buckets/michaelsmith6/U-SEG-Models) and [mirrored at McGill University](https://library.cim.mcgill.ca/data/models/U-SEG/).

## Datasets
See [the README](datasets/README.md).

## Getting Started

The basic process for training and model if necessary (as we do for datasets such as VIPER) and running inference follows the standard Mask2Former format, documented [on their page](https://github.com/facebookresearch/Mask2Former/blob/main/GETTING_STARTED.md).


## Configs
All of the predictions made by Mask2Former models are done through configs in the `configs` directory. They are in one of four directories:
- `cityscapes/panoptic-segmentation`
- `cityscapes_VIPERformat/panoptic-segmentation`
- `viper/panoptic-segmentation`
- `VIPER_cityscapesformat/panoptic-segmentation`

All others are inherited from Mask2Former.

## General Procedure
To generate the results used in the paper, the general procedure is:

1. Download the datasets and set them up using the scripts in the datasets folder and listed in the [datasets README](datasets/README.md).
2. Acquire the weights or train a model. For example, with Cityscapes, we used the model trained by the Mask2Former authors, but with VIPER we needed to train a model ourselves. This was done via the following command:
    ```bash
    python train_net.py --num-gpus="${SLURM_GPUS_ON_NODE}" --config-file "${MASK2FORMER_CONFIG_FILE}" OUTPUT_DIR ${OUTPUT_DIR}
    ```
    An example config file for training is `configs/viper/panoptic-segmentation/maskformer2_R50_bs1_300ep.yaml`. Another example is `configs/viper/panoptic-segmentation/timm/maskformer2_vit-b_dinov2_bs1_300ep.yaml`, which loads a pretrained DINOv2 ViT-B model from timm and then trains from there.
3. Run inference on the dataset e.g.
    ```bash
    python train_net.py --eval-only --num-gpus="${SLURM_GPUS_ON_NODE}" --config-file "${MASK2FORMER_CONFIG_FILE}" OUTPUT_DIR "${OUTPUT_DIR}$" SEED "${SLURM_ARRAY_TASK_ID}"
    ```
    Relevant config files can be found in the `uncert_eval` folder e.g. `configs/viper/panoptic-segmentation/uncert_eval/r50/maskformer2_R50_MC3_maskdist.yaml` for 3 MC Dropout samples. Effectively all experiment settings are defined in these config files, and they can inherit from each other.
    
    Running inference with different models, datasets, and seeds will generate a lot of data, especially as we save a lot of "raw" results to disk via the `saveResults` "evaluator" class. Some metrics, such as those for calibration, are calculated on the fly. Others, however, including those for out-of-distribution detection and failure detection, are calculated later (we call this offline) as they are somewhat slow to calculate and would waste time better spent sending new images to the GPU. Thus, we save the data we need for these calculations, such as per-pixel predictions and uncertainty information, to disk. Do note that this will very quickly consume terabytes of disk space, as there are multiple models, datasets, seeds and other variables.
4. These offline calculations are done using the `offline_preprocess.py` script:
    ```bash
    python offline_preprocess.py "${EXPERIMENTS_FOLDER}" --delete --yes
    ```
    which, with the option --delete specified, will process the potentially terabytes of data, save the processed results to disk, and delete the original files.
5. Once the offline processing is done, there is still a lot of manipulation required to create the figures in the paper. These are all done using Jupyter notebooks in the `notebooks` folder.

## Notebooks
The notebooks are used for the final figure generation for the paper. Much of the analysis that resulted in the paper were done while writing them, so they are disorganized but are provided for completeness for anyone building off this work.
- `process_results.ipynb` - The main notebook for most of the plots in the paper. Run this first as it generates files used by some other ones.
- `process_results_ood.ipynb` - Similar to the main notebook, but for the OOD experiments. Run this after `process_results.ipynb`.
- `process_aggcompare_results.ipynb` - Handles the comparison of different aggregation methods supporting the use in the paper of the introduced Mask Distance method.
- `diagram.ipynb` - Used to get some prediction images for the overview diagram in the paper.
- `timing.ipynb` - Used to calculate some of the runtime comparison results in the Supplementary Material.

We also provide CSV versions of the DataFrames created in the notebooks; these contain all the key information but are significantly more compact that even the processed results from which they are derived. Simply find the cells where they are saved (e.g. `df.to_csv('all_uncert_data_exceptood.csv')`) and load them instead.
- `aggcompare_results.csv` - The DataFrame used in `process_aggcompare_results.ipynb`.
- `all_uncert_data_exceptood.csv` - The main DataFrame used in `process_results.ipynb`. 
- `uncert_data_ood_only.csv` - The DataFrame used in `process_results_ood.ipynb`.

## Citation

If you make use of any code, data or results, please cite as follows:

```BibTeX
@InProceedings{Smith_2026_CVPRF,
    author    = {Smith, Michael and Ferrie, Frank},
    title     = {U-SEG: Uncertainty in SEGmentation - A systematic multi-variable exploration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026}
}
```
You may also want to [cite the original Mask2Former paper](https://github.com/facebookresearch/mask2former) if appropriate.
