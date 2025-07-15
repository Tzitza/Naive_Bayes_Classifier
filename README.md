# Naive Bayes Classifier (NBC)

## Overview

This project implements a Naive Bayes Classifier in Python that can handle both discrete and continuous attributes. The classifier is designed to work with CSV datasets and provides functionality for training, prediction, and evaluation.

## Features

- **Mixed Data Types**: Handles both discrete (D) and continuous (C) attributes
- **Gaussian Distribution**: Uses Gaussian distribution for continuous attributes
- **Laplace Smoothing**: Implements Laplace smoothing to handle zero frequency problems
- **Interactive Queries**: Allows users to input custom queries for classification
- **Model Evaluation**: Includes accuracy calculation on test data
- **Parameter Visualization**: Displays all learned model parameters

## Requirements

- Python 3.x
- pandas
- scikit-learn
- math (built-in)
- collections (built-in)

## Installation

```bash
pip install pandas scikit-learn
```

## Dataset Format

The classifier expects CSV files with the following format:

### Header Format
```
"attribute1",TYPE,"attribute2",TYPE,...,"class"
```

Where TYPE is either:
- `C` for continuous attributes
- `D` for discrete attributes

### Example (Iris Dataset)
```
"sepal.length",C,"sepal.width",C,"petal.length",C,"petal.width",C,"variety"
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
...
```

## Usage

### Basic Usage

```python
from NBC import NaiveBayesClassifier

# Create classifier instance
nb = NaiveBayesClassifier()

# Train the model
test_data = nb.train("IRIS.csv")

# Print model parameters
nb.print_params()

# Interactive query
query = nb.get_user_query()
probabilities = nb.predict_proba(query)
```

### Running the Complete Program

```bash
python NBC.py
```

## Class Structure

### NaiveBayesClassifier

#### Attributes
- `classes`: List of unique class labels
- `attr_types`: Dictionary mapping attribute names to their types (C/D)
- `class_prob`: Dictionary of class prior probabilities P(c)
- `attr_params`: Dictionary storing parameters for each attribute-class combination
- `laplace_alpha`: Laplace smoothing parameter (default: 1.0)

#### Methods

- `load_data(file_path)`: Loads and preprocesses CSV data
- `train(file_path)`: Trains the classifier and returns test data
- `predict_proba(input_data)`: Predicts class probabilities for given input
- `print_params()`: Displays all learned model parameters
- `get_user_query()`: Interactive method for user input

## Algorithm Details

### Training Process

1. **Data Loading**: Parses CSV file and extracts attribute types
2. **Data Splitting**: Uses 70-30 stratified split for training/testing
3. **Prior Calculation**: Computes P(c) for each class
4. **Likelihood Estimation**:
   - **Continuous attributes**: Estimates μ and σ for Gaussian distribution
   - **Discrete attributes**: Counts occurrences with Laplace smoothing

### Prediction Process

1. **Prior Initialization**: Starts with P(c) for each class
2. **Likelihood Multiplication**: 
   - **Continuous**: Uses Gaussian probability density function
   - **Discrete**: Uses stored probability values
3. **Normalization**: Converts to proper probability distribution

### Mathematical Formulation

#### For Continuous Attributes:
```
P(x|c) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
```

#### For Discrete Attributes (with Laplace Smoothing):
```
P(x|c) = (count(x,c) + α) / (count(c) + α * |values|)
```

Where:
- `α` = Laplace smoothing parameter
- `|values|` = number of unique values for the attribute

## Example Output

```
Πιθανότητες Κλάσεων P(c):
Iris-setosa: 0.3333
Iris-versicolor: 0.3333
Iris-virginica: 0.3333

Παράμετροι Χαρακτηριστικών:

sepal.length (C):
Iris-setosa: μ = 5.01, σ = 0.35
Iris-versicolor: μ = 5.94, σ = 0.52
Iris-virginica: μ = 6.59, σ = 0.64

Posterior Probabilities:
P(Iris-setosa|x) = 0.000001
P(Iris-versicolor|x) = 0.999998
P(Iris-virginica|x) = 0.000001

Accuracy on test set: 95.56%
```

## File Structure

```
project/
├── NBC.py              # Main classifier implementation
├── IRIS.csv           # Example dataset
└── README.md          # This file
```

## Key Features Implementation

### Zero Frequency Problem
- Handled using Laplace smoothing with α = 1.0
- Prevents zero probabilities for unseen attribute values

### Gaussian Distribution
- Continuous attributes modeled using normal distribution
- Parameters (μ, σ) estimated from training data
- Minimum standard deviation of 0.01 to prevent numerical issues

### Stratified Sampling
- Ensures balanced class distribution in train/test splits
- Uses scikit-learn's `train_test_split` with `stratify` parameter

## Error Handling

- **Input Validation**: Checks for proper data types during user input
- **Numerical Stability**: Prevents division by zero in standard deviation
- **Missing Values**: Drops rows with NaN values during data loading
- **Unknown Attributes**: Gracefully handles attributes not seen during training

## Performance Considerations

- **Memory Efficient**: Uses defaultdict for sparse parameter storage
- **Vectorized Operations**: Leverages pandas for efficient data processing
- **Scalable**: Linear complexity in number of training examples

## Limitations

- Assumes feature independence (naive assumption)
- Gaussian assumption may not hold for all continuous attributes
- Performance depends on quality of training data distribution

## Future Enhancements

- Support for missing values during prediction
- Cross-validation for better performance estimation
- Feature selection capabilities
- Support for different probability distributions
- Batch prediction functionality

## License

This project is developed for educational purposes as part of a Machine Learning course assignment.

## Author

Developed in Python environment, chosen for its extensive machine learning libraries and ease of implementation for algorithms like Naive Bayes Classifier.
