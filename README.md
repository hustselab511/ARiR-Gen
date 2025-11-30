# ARiR-Gen: an Autoregression-Induced Residual Generative Approach for Robust ECG Denoising
This repository provides a PyTorch implementation of Autoregression-Induced Residual Generation(ARiR-Gen),which combines two stages to improve the generalizability and interpretability of ECG signal denoising.

## Code structure
* `configs`: Configuration files for different model implementations, including training hyperparameters, dataset settings, and experiment setups.
* `datasets`: Dataset wrappers for **QT** and **SimEMG** Database. This module also provides different data downloading and loading functions
* `utils`: Utility functions for training and evaluation.
* `models`: Model definitions for ARiR-Gen and baselines.
    * `digitial_filters`:Implementations of digitiall signal processing filters, including **FIR** and **IIR** designs.
    * `dl_filters`:Definitions of deep-learning based filtering models.
    * `generation_filters`:Definitions of generative models,including **ARiR-Gen**.
* `eval.py`:Evaluation script for running experiments on trained models. It loads saved checkpoints, computes quantitative metrics, and stores evaluation results.
* `main.py`:The main entry point for training.
* `trainer.py`:Provides the concrete training routines for different methods.

## Setup
### Clone Repository
```
git clone https://github.com/hustselab511/ARiR-Gen.git
cd ARiR-Gen
```
### Environment
We use conda to manage the environment. The installation is tested with CUDA 11.8 and NVIDIA RTX 4090(24GB).
```
conda create -n arirgen python=3.10 -y
conda activate arirgen
pip install -r requirements.txt
```
### Data Preparation
We provide standard procedures for downloading and preprocessing [MIT-BIH Noise Stress Test Database](https://physionet.org/content/nstdb/1.0.0/), [QT Database](https://physionet.org/content/qtdb/1.0.0/) and [SimEMG database](https://data.mendeley.com/datasets/yx5pb66hwz/1).For other datasets, you may optionally download them and preprocess them in any way that best fits the model input format and downstream evaluation tasks.
To download and prepare the above datasets, run:

```
mkdir -p data
cd datasets/scripts
source download_nst_and_qt.sh
source download_simemg.sh
```
After completing the data preparation step, your **data/** directory should have the following structure:
```
data/
    ├── MITBIH-NSTDB/
    │   ├── bw.dat
    │   ├── em.dat
    │   ├── ma.dat
    │	└──...
    └── QT/
    │   ├── records/
    │   │   ├── *.dat
    │   │   ├── *.hea
    │   │   ├── *.pu1
    │	│	└──...
    │   ├── noise.pkl
    │   └── data.pkl
    └── SimEMG/
    	├── records/
    	│	└── *.mat
    	└── data.pkl
```

## Train the model
### Train first-stage model
```
python main.py --exp_name AR --n_type 1 --use_rmn
python main.py --exp_name AR --n_type 2 --use_rmn
```
### Train second-stage model
```
python main.py --exp_name ARiRGen --n_type 1 --use_rmn
python main.py --exp_name ARiRGen --n_type 2 --use_rmn
```

## Evaluate the model
```
python eval.py --exp_name ARiRGen --dataset QT --type 2 --use_rmn
python eval.py --exp_name ARiRGen --dataset SimEMG --type 2
```

## Acknowledgements

The data preprocessing and evaluation pipeline and model implementations are based on some open-source codes.We appreciate the great help of these works.
- [DeepFilter](https://github.com/fperdigon/DeepFilter)
- [Score-based-ECG-Denoising](https://github.com/HuayuLiArizona/Score-based-ECG-Denoising/tree/main)
- [TCDAE](https://github.com/chchenmeng/ECG-denosing-by-TCDAE)
- [RDDM](https://github.com/nachifur/RDDM)