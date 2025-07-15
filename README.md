# Naive Bayes Classifier (NBC) in Python

This repository contains a custom implementation of a **Naive Bayes Classifier (NBC)** from scratch in Python, designed to support **both discrete (D)** and **continuous (C)** attributes using the **Gaussian distribution** for the latter.

---

## 📁 Files

- `NBC.py`: The main Python script containing the Naive Bayes Classifier.
- `IRIS.csv`: Example dataset (assumed) formatted with a header specifying the attribute types.

---

## ✅ Features

- Handles both discrete and continuous attributes.
- Automatically computes:
  - Prior probabilities for each class.
  - Conditional probabilities with **Laplace smoothing**.
  - Mean and standard deviation for Gaussian-distributed features.
- Accepts user queries for prediction.
- Displays posterior probabilities.
- Splits dataset into training (70%) and testing (30%) using stratified sampling.
- Calculates and displays model accuracy on test data.

---

## 📥 Input Format

CSV file must follow the below format:

```csv
"feature1",C,"feature2",D,...,"class"
value1,value2,...,"ClassLabel"
...
