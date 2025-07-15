# Naive Bayes Classifier for Iris Dataset

This repository contains a Python implementation of a Naive Bayes Classifier (NBC) designed to classify the Iris dataset. The classifier handles both continuous (Gaussian-distributed) and discrete attributes, incorporates Laplace smoothing to address zero-frequency problems, and provides probabilistic predictions for user queries.

## Features

- **Data Handling**: Loads CSV files with mixed attribute types (continuous/discrete) and automatically detects their distributions.
- **Training**: Computes class probabilities (priors) and likelihoods for Gaussian and discrete attributes.
- **Prediction**: Calculates posterior probabilities for user-provided queries.
- **Evaluation**: Includes a test set accuracy metric to validate performance.
- **User Interaction**: Allows interactive input for custom queries.

## Requirements

- Python 3.x
- pandas
- scikit-learn (for data splitting)

## Usage

1. **Training the Model**:  
   Run the script to train the model on the included `IRIS.csv` dataset:
   ```bash
   python NBC.py
