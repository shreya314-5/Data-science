# Predictive Modeling Using Machine Learning

## Project Overview

This project demonstrates a complete supervised machine-learning workflow for predicting outcomes from structured data.

The **Breast Cancer Wisconsin Diagnostic dataset** available through scikit-learn is used as the example dataset. The target is binary classification:

- `0` = malignant
- `1` = benign

The project compares three classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest

## Objectives

- Load and inspect a real-world dataset
- Separate features and target
- Split data into training and testing sets
- Train multiple supervised-learning models
- Evaluate predictions using standard classification metrics
- Visualize model performance with confusion matrices
- Compare ROC curves and ROC-AUC values
- Save results that can be inspected directly from the repository

## Project Structure

```text
predictive-modeling-ml/
│
├── predictive_modeling.py
├── breast_cancer_data.csv
├── requirements.txt
├── README.md
└── results/
    ├── logistic_regression_confusion_matrix.png
    ├── decision_tree_confusion_matrix.png
    ├── random_forest_confusion_matrix.png
    ├── model_comparison.csv
    ├── model_comparison.png
    └── roc_curves.png
```

## Machine Learning Workflow

### 1. Data Preparation

The dataset is loaded with `sklearn.datasets.load_breast_cancer()`.

The data is divided into:

- **80% training data**
- **20% testing data**

A fixed `random_state=42` is used so that the experiment is reproducible.

### 2. Models

#### Logistic Regression

Logistic Regression is used as a baseline classification model. Feature standardization is applied before training.

#### Decision Tree

A Decision Tree learns a sequence of feature-based decisions. The tree depth is limited to reduce overfitting.

#### Random Forest

Random Forest combines multiple decision trees and averages their predictions. The implementation uses 200 trees.

## Evaluation Metrics

The project calculates:

- **Accuracy** – proportion of all predictions that are correct
- **Precision** – proportion of predicted positive cases that are actually positive
- **Recall** – proportion of actual positive cases correctly identified
- **F1 Score** – harmonic mean of precision and recall
- **ROC-AUC** – measures how well the model separates the two classes across classification thresholds

It also generates:

- Confusion matrices
- Model comparison chart
- ROC curves

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd predictive-modeling-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python predictive_modeling.py
```

The script prints classification reports in the terminal and creates the result files inside the `results/` folder.

## Sample Result Interpretation

The generated `model_comparison.csv` contains the measured performance of all three models.

The ROC graph allows the models to be visually compared by their true-positive and false-positive rates.

The confusion matrices show the numbers of correct and incorrect predictions for each class.

## Key Learning Outcomes

By completing this project, the following machine-learning concepts are demonstrated:

- Supervised learning
- Binary classification
- Training and testing datasets
- Feature scaling
- Logistic Regression
- Decision Trees
- Random Forest
- Model evaluation
- Confusion matrices
- ROC curves
- ROC-AUC
- Reproducible experiments

## Important Note

This repository is an educational machine-learning project. The dataset is publicly available through scikit-learn and is used here to demonstrate the modeling workflow. The model should not be treated as a clinical diagnostic system.

## Author

**Shreya**
