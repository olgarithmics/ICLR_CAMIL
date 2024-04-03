# CAMIL: Context-Aware Multiple Instance Learning for Cancer Detection and Subtyping in Whole Slide Images (ICLR 2024 spotlight)
This is the implementation of [CAMIL](https://arxiv.org/abs/2305.05314) (work in progress)

![Pipeline](assets/pipeline.png "An overview of the CAMIL model architecture. First, WSIs are preprocessed to separate tissue
from the background. Then, the WSIs are split into fixed-size tiles of size 256 × 256 and fed through a pre-
trained feature extractor to obtain feature representations of size 1024 for each tile. A Nystromformer module
then transforms these feature embeddings. These transformed feature embeddings are then used as input to
our neighbor-constrained attention module. This module allows attending over each patch and its neighboring
patches, generating a neighborhood descriptor of each tile’s closest neighbors, and calculating their attention
coefficients. The output layer then aggregates the tile-level attention scores produced in the previous layer to
emit a final slide classification score.")

Abstract:
> The visual examination of tissue biopsy sections is fundamental for cancer diagnosis, with pathologists analyzing sections at multiple magnifications to discern tumor cells and their subtypes. However, existing attention-based multiple instance learning (MIL) models used for analyzing Whole Slide Images (WSIs) in cancer diagnostics often overlook the contextual information of tumor and neighboring tiles, leading to misclassifications. To address this, we propose the Context-Aware Multiple Instance Learning (CAMIL) architecture. CAMIL incorporates neighbor-constrained attention to consider dependencies among tiles within a WSI and integrates contextual constraints as prior knowledge into the MIL model. We evaluated CAMIL on subtyping non-small cell lung cancer (TCGA-NSCLC) and detecting lymph node (CAMELYON16 and CAMELYON17) metastasis, achieving test AUCs of 97.5\%, 95.9\%, and 88.1\%, respectively, outperforming other state-of-the-art methods. Additionally, CAMIL enhances model interpretability by identifying regions of high diagnostic value.

## WSI pre-processing
we follow the CLAM's WSI processing solution (https://github.com/mahmoodlab/CLAM)

```python
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --preset bwh_biopsy.csv --seg --patch --stitch
```


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
