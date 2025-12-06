# PFAS-AEDL

This is the code for "Adaptive Evidential Deep Learning Framework for Predicting PFAS Environmental Transport Properties" paper.

## Directory Structure

```shell
/
├── ML/
│   ├── DT.py
│   ├── RF.py
│   └── XGB.py
├── chemprop/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── scaffold.py
│   │   ├── scaler.py
│   │   └── utils.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── features_generators.py
│   │   ├── featurization.py
│   │   └── utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── mpn.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── confidence_estimator.py
│   │   ├── confidence_evaluator.py
│   │   ├── cross_validate.py
│   │   ├── evaluate.py
│   │   ├── make_predictions.py
│   │   ├── predict.py
│   │   ├── run_training.py
│   │   └── train.py
│   ├── __init__.py
│   ├── nn_utils.py
│   ├── parsing.py
│   └── utils.py
├── configs/
│   ├── config.json
│   ├── pfas_KAW_best.json
│   ├── pfas_KOA_best.json
│   ├── pfas_KOC_best.json
│   ├── pfas_KOW_best.json
│   └── pfas_W_best.json
├── data/
│   └── SI-2.xlsx
├── features/
│   ├── KAW_count.npz
│   ├── KAW_morgan.npz
│   ├── KAW_pfas.npz
│   ├── KAW_pfas_combined.npz
│   ├── KOA_count.npz
│   ├── KOA_morgan.npz
│   ├── KOA_pfas.npz
│   ├── KOA_pfas_combined.npz
│   ├── KOC_count.npz
│   ├── KOC_morgan.npz
│   ├── KOC_pfas.npz
│   ├── KOC_pfas_combined.npz
│   ├── KOW_count.npz
│   ├── KOW_morgan.npz
│   ├── KOW_pfas.npz
│   ├── KOW_pfas_combined.npz
│   ├── W_count.npz
│   ├── W_morgan.npz
│   ├── W_pfas.npz
│   └── W_pfas_combined.npz
├── scripts/
│   ├── bench_conf.py
│   ├── bench_figs.py
│   ├── molecular_split.py
│   ├── get_bench.py
│   ├── get_features.py
│   └── results_figures.py
├── LICENSE
├── READM.md
├── hyperparameter_optimization.py
├── logs.zip
├── resultss.zip
├── splits.zip
└── train.py
```

##  Overview

This repository implements an **Adaptive Evidential Deep Learning (AEDL)** framework specifically designed for predicting environmental transport properties of Per- and Polyfluoroalkyl Substances (PFAS). The framework combines molecular graph neural networks with evidential deep learning to provide both accurate predictions and reliable uncertainty quantification.

### Built on Evidential Deep Learning

> **Note**: This implementation extends the [aamini/chemprop](https://github.com/aamini/chemprop) evidential deep learning framework with adaptive loss components and PFAS-specific features.

##  Installation

### Prerequisites
```bash
 Python 3.7 or higher
```
### Install Dependencies
```bash
# Clone the repository
git clone https://github.com/xysonglab/PFAS-AEDL.git
cd PFAS-AEDL
```
### Create conda environment
```bash
conda env create -f environment.yml
```
## Dataset

This study obtained 4,519 PFAS compound records from the NORMAN website (https://www.norman-network.com/nds/susdat/).

```bash
# Extract Molecular Features
python get_features.py \
    --data_path data/[dataset.csv] \
    --features_generator [pfas] \
    --save_path features/[pfas_features.npz]
  --
```
## Train a Model
```bash
# Training with adaptive regularization
python train.py \
  --data_path data/pfas_dataset.csv \
  --save_dir results/pfas_adaptive \
  --feature_path features/[features].npz \
  --confidence evidence \
  --new_loss \
  --use_adaptive_reg True \
  --use_robust_loss True \
  --use_calibration True \
  --regularizer_coeff 0.2 \
  --
```

### Benchmarking Multiple Methods
```bash
python_get_bench.py [config.json]
```
## ML
```bash
python XGB.py \ 
--csv_file [pfas_dataset.csv] \ 
--target_column [property] \ 
--output_dir  ./results/XGB/pfas_dataset/[features].npz \ 
--features_generator [features]
```
