# MOMA: A unified feature-motion consistency framework for robust image matching

<br/>

## Installation
For environment and data setup, please refer to [LoFTR](https://github.com/zju3dv/LoFTR).


## Run MOMA

### Download Datasets
You need to setup the testing subsets of MegaDepth first from [driven](https://drive.google.com/drive/folders/1TE_zJlKfPFRLeIrtq5iMBBjg-XaovNon).

For the data utilized for training, we use the same training data as [LoFTR](https://github.com/zju3dv/LoFTR) does.


### Megadepth validation
For different scales, you need edit [megadepth_test_1500](configs/data/megadepth_test_1500.py).

```shell
# with shell script
bash ./scripts/reproduce_test/outdoor_moma.sh
```

<br/>


## Training
We train Moma on the MegaDepth datasets following [LoFTR](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md). And the results can be reproduced when training with 32gpus. Please run the following commands:

```
sh scripts/reproduce_train/outdoor.sh
```
## Acknowledgement

This repository is developed from `LoFTR`, `Adamatcher`, `AspanFormer`, and we are grateful to its authors for their [implementation](https://github.com/zju3dv/LoFTR).

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
}
```
