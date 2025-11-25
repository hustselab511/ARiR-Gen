# ARiR-Gen: an Autoregression-Induced Residual Generative Approach for Robust ECG Denoising
This repository provides a PyTorch implementation of Autoregression-Induced Residual Generation(ARiR-Gen),which combines two stages to improve the generalizability and interpretability of ECG signal denoising.

## Code structure
* `configs`: Configuration files for different model implementations, including training hyperparameters, dataset settings, and experiment setups.
* `datasets`: Dataset wrappers for **QT**, **SimEMG**, and the **MIT-BIH Arrhythmia** Database. This module also provides different data loading functions
* `utils`: Utility functions for training and evaluation.
* `models`: Model definitions for ARiR-Gen and baselines.
    * `digitial_filters`:Implementations of digitiall signal processing filters, including **FIR** and **IIR** designs.
    * `dl_filters`:Definitions of deep-learning based filtering models.
    * `generation_filters`:Definitions of generative models,including **ARiR-Gen**.
* `eval.py`:Evaluation script for running experiments on trained models. It loads saved checkpoints, computes quantitative metrics, and stores evaluation results.
* `main.py`:The main entry point for training.
* `trainer.py`:Provides the concrete training routines for different methods.

## Setup
Install the package via
```
pip install -r requirements.txt
```

## Train the model
```
python main.py --exp_name ARiRGen --n_type 1
```
```
python main.py --exp_name ARiRGen --n_type 2
```

## Evaluate the model
```
python eval.py --exp_name ARiRGen --dataset QT --type 2
```

## Acknowledgements

The data preprocessing and evaluation pipeline and model implementations are based on some open-source codes.We appreciate the great help of these works.
- [DeepFilter](https://github.com/fperdigon/DeepFilter)
- [Score-based-ECG-Denoising](https://github.com/HuayuLiArizona/Score-based-ECG-Denoising/tree/main)
- [TCDAE](https://github.com/chchenmeng/ECG-denosing-by-TCDAE)
- [RDDM](https://github.com/nachifur/RDDM)