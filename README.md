# CAMIL: Context-Aware Multiple Instance Learning for Cancer Detection and Subtyping in Whole Slide Images (ICLR 2024 spotlight)
Tensorflow implementation for the multiple instance learning model described in the paper [CAMIL](https://arxiv.org/abs/2305.05314) (work in progress)

![Pipeline](assets/pipeline.png "An overview of the CAMIL model architecture. First, WSIs are preprocessed to separate tissue
from the background. Then, the WSIs are split into fixed-size tiles of size 256 × 256 and fed through a pre-
trained feature extractor to obtain feature representations of size 1024 for each tile. A Nystromformer module
then transforms these feature embeddings. These transformed feature embeddings are then used as input to
our neighbor-constrained attention module. This module allows attending over each patch and its neighboring
patches, generating a neighborhood descriptor of each tile’s closest neighbors, and calculating their attention
coefficients. The output layer then aggregates the tile-level attention scores produced in the previous layer to
emit a final slide classification score.")

## Installation
Two separate conda environments are used for different stages of the workflow:
- `dl_torch` for WSI preprocessing and feature extraction
- `alma` for model training and evaluation

To create and activate these environments, run the following commands:
```bash
$ conda env create --name torch_env --file torch_env.yml
$ conda activate torch_env
```
## WSI preprocessing
we follow the CLAM's WSI processing solution (https://github.com/mahmoodlab/CLAM)

```bash
# WSI Segmentation and Patching
python create_patches_fp.py --source DATA_DIRECTORY --save_dir PATCHES_RESULTS_DIRECTORY --patch_size 256 --preset bwh_biopsy.csv --seg --patch --stitch
```

## Feature extraction
This script computes features using pre-trained weights and saves the results in the specified output directory
```bash
# WSI Segmentation and Patching
python feature_extractor/compute_feats.py  --weights weight_dir/*.pth  --dataset "PATCHES_RESULTS_DIRECTORY/*" --output FEAT_RESULTS_DIRECTORY --slide_dir DATA_DIRECTORY 
```
- `weight_dir`: Directory containing checkpoints, one from the TCGA-NSCLC and the other from Calmeyon-16.
- `output_dir`: Directory where the H5 files are stored.
- `slide_dir`: Directory where the slides are stored.


## Train
This script computes features using pre-trained weights and saves the results in the specified output directory
```bash
python run.py --experiment_name EXP_NAME --epoch 30 --feature_path FEAT_RESULTS_DIRECTORY --label_file LABEL_FILE --csv_file SPLIT_DIR --save_dir WEIGHT_DIR
```
-  `experiment_name`: The name of the experiment.
-  `epoch`: The number of training epochs.
-  `feature_path`: The path where the features are stored.
-  `label_file`: The path to the CSV file containing labels.
-  `csv_file`: The path to the CSV file containing data splits.
-  `save_dir`: The directory where the weights are saved stored.
- 
The label file should be a CSV file with the following comma-separated fields:
case_id, slide_id, label, slide_label


## Test
This script computes features using pre-trained weights and saves the results in the specified output directory
```bash
python run.py --experiment_name EXP_NAME  --test  --feature_path FEAT_RESULTS_DIRECTORY --label_file LABEL_FILE --csv_file SPLIT_DIR --save_dir WEIGHT_DIR
```
-  `experiment_name`: The name of the experiment.
-  `feature_path`: The path where the features are stored.
-  `label_file`: The path to the CSV file containing labels.
-  `csv_file`: The path to the CSV file containing data splits.
-  `save_dir`: The directory where the weights are saved stored.
-  `test`: Flag indicating the test stage 



If you use this code, please cite our work using:
```bibtex
@inproceedings{
fourkioti2024camil,
title={{CAMIL}: Context-Aware Multiple Instance Learning for Cancer Detection and Subtyping in Whole Slide Images},
author={Olga Fourkioti and Matt {De Vries} and Chris Bakal},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=rzBskAEmoc}
}
```
