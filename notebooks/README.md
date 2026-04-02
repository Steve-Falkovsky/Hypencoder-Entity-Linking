# Hypencoder for Biomedical Entity Linking

This repository contains the Jupyter notebooks used to reproduce the experiments presented in the dissertation.

The notebooks implement training and evaluation of SapBERT Bi-Encoder and Hypencoder models for biomedical entity linking.

## Contents

- Jupyter notebooks for:
  - Data loading
  - Model training
  - Evaluation

- Scripts (invoked within notebooks):
  - Environment setup
  - Data preprocessing

## Notebook Overview

- `Bi-encoder_fine-tune.ipynb`
  - Fine tunes SapBERT on BC5CDR data including context

- `Bi-encoder_experiments_evaluation.ipynb`
  - Evaluates the fine-tuned model on several metrics and saves results to file

- `Hypencoder_training.ipynb`
  - Trains Hypencoder models with different configurations

- `Hypencoder_experiments_evaluation.ipynb`
  - Evaluates the hypencoder models on several metrics and saves results to file

## Running Experiments and Reproducing Results

The notebooks are designed to run in Google Colab:

1. Open the relevant notebook in Google Colab
2. Select appropriate model (details provided in notebooks and `manual.md`)
3. Run all cells
4. Models and data will be loaded from HuggingFace automatically (No manual data download is required).
