# RecTable: Fast Modeling Tabular Data with Rectified Flow

<p align="center">
  <a href="https://github.com/fmp453/rectable/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/badge/license-Apache 2.0-green">
  </a>
  <a href="https://arxiv.org/abs/2503.20731">
    <img alt="arXiv Url" src="https://img.shields.io/badge/cs.LG-2503.20731-B31B1B.svg">
  </a>
</p>

This repository contains the implementation of [RecTable: Fast Modeling Tabular Data with Rectified Flow](https://arxiv.org/abs/2503.20731).


## Installation
The environments of our experiments are based on PyTorch 2.3.1 (docker image)

Pull docker image.

```bash
docker pull pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
```

Install other packages in the container.
```bash
pip install -r requirements.txt
```

## Preparing Datasets
### Using the datasets experimented in the paper
Download raw dataset:

```bash
python download_dataset.py
```

Process dataset:
```bash
python process_dataset.py
```

First, create a directory for you dataset [NAME_OF_DATASET] in ./data:
```
cd data
mkdir [NAME_OF_DATASET]
```

Put the tabular data in .csv format in this directory ([NAME_OF_DATASET].csv). **The first row should be the header** indicating the name of each column, and the remaining rows are records.

Then, write a .json file ([NAME_OF_DATASET].json) recording the metadata of the tabular, covering the following information:
```
{
    "name": "[NAME_OF_DATASET]",
    "task_type": "[NAME_OF_TASK]", # binclass or regression
    "header": "infer",
    "column_names": null,
    "num_col_idx": [LIST],  # list of indices of numerical columns
    "cat_col_idx": [LIST],  # list of indices of categorical columns
    "target_col_idx": [list], # list of indices of the target columns (for MLE)
    "file_type": "csv",
    "data_path": "data/[NAME_OF_DATASET]/[NAME_OF_DATASET].csv"
    "test_path": null,
}
```
Put this .json file in the .Info directory.

Finally, run the following command to process the UDF dataset:

```bash
python process_dataset.py --dataname [NAME_OF_DATASET]
```

## Training 

```bash
python main.py --dataname [NAME_OF_DATASET] --exp "rectable_[NAME_OF_DATASET]"
```

Other configurations can be found in `config.py`.

## Generation

```bash
python generate.py --ckpt CKPT_PATH --dataname [NAME_OF_DATASET] 
```

`CKPT_PATH` is same of `exp` in the configuration when training. For example, when the checkpoint's path is `checkpoints/rectable_sample.pth`, the generation command as follows:
```bash
python generate.py --ckpt sample --dataname [NAME_OF_DATASET]
```


## Acknowledgement
This repo is built upon the previous work [TabSyn's](https://github.com/amazon-science/tabsyn), [TabDiff's](https://github.com/MinkaiXu/TabDiff), and [Rectified Flow](https://github.com/gnobitab/RectifiedFlow). Many thanks to Hengrui, Shi, and Liu!

## Citation
We appreciate your citations if you find this repository useful to your research!

```bibtex
@misc{fuchi2025rectablefastmodelingtabular,
      title={RecTable: Fast Modeling Tabular Data with Rectified Flow}, 
      author={Masane Fuchi and Tomohiro Takagi},
      year={2025},
      eprint={2503.20731},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.20731}, 
}
```

## Contact
The email in the paper is not availabe due to author's graduation. Please contact us via Twitter (X). The account can be found at [my profile page](https://github.com/fmp453).
