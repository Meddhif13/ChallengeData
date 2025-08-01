# ChallengeData

This repository contains my solution template for a MathA **ChallengeData** machine‑learning competition. The goal of the challenge is to build a model that predicts two target variables from 11 input features. The provided training set contains over six million samples which makes efficient data loading and a clean project structure essential.

## Problem Statement
Participants receive three CSV files:

- `x_train.csv` – feature matrix for training (6M​+ rows, 11 columns)
- `y_train.csv` – one‑hot encoded targets for the training set
- `x_test.csv` – feature matrix for which predictions must be submitted

The task is to train a model on `x_train`/`y_train` and use it to predict the labels for `x_test`.

## Approach
1. **Data loading** – implemented in `data_loader.py`. The script reads the CSV files using pandas and can be invoked standalone or imported as a module.
2. **Model** – a small feed‑forward network implemented in TensorFlow/Keras. Training logic lives in `train.py` and uses an 80/20 split of the training data for validation.
3. **Evaluation** – `evaluate.py` reloads the saved model and reports validation accuracy.

The helper scripts expose functions so that they can be imported from a notebook or executed from the command line.

## Results
The provided baseline model is intentionally simple. Training for a few epochs yields a validation accuracy around 70% on the supplied data split. This serves as a starting point for further experimentation.

## Installation
1. Install Python 3.10 or later.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage
Example commands to reproduce the workflow:

```bash
# Inspect the data shapes
python data_loader.py --data-folder ./data

# Train a model (saved as model.h5)
python train.py --data-folder ./data --epochs 5 --model-path model.h5

# Evaluate the trained model
python evaluate.py --data-folder ./data --model-path model.h5
```

A small demonstration notebook `Demo.ipynb` shows how to call the same functions from a Jupyter environment.

## Future Work
- Hyper‑parameter tuning and model architecture search
- More sophisticated feature engineering
- Cross‑validation and automated reporting

## Repository Structure
```
.
├── data_loader.py    # data loading utilities
├── train.py          # training script
├── evaluate.py       # evaluation script
├── Demo.ipynb        # demo notebook
├── requirements.txt  # project dependencies
└── .gitignore        # ignores data and model artifacts
```
