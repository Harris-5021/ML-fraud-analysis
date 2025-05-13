Machine Learning Fraud Detection Systems
This repository contains three machine learning systems developed for fraud detection and image classification. Each system demonstrates different approaches to machine learning, including supervised learning, unsupervised learning, and convolutional neural networks.
Project Overview
This project was developed as part of a Machine Learning Algorithms and Heuristics module to showcase:

Data preprocessing and exploratory data analysis
Implementation of various machine learning algorithms
Model evaluation and comparison
Image classification using CNNs
Fraud detection using both supervised and unsupervised learning

Contents
The repository includes three main Python scripts:

heuristics.py: A comprehensive fraud analysis system with various machine learning algorithms
credit_card.py: Credit card fraud detection using multiple classification models
cnn.py: Image classification system using CNNs with the Intel Image Dataset

1. Fraud Analysis System (heuristics.py)
This script implements a complete fraud analysis system with a user-friendly interface that allows:
Features:

Data loading, cleaning, and preprocessing
Exploratory data analysis with visualizations
Feature correlation analysis
Statistical tests for categorical relationships with fraud
Training and evaluation of multiple supervised learning models:

Logistic Regression
Random Forest
Decision Tree
K-Nearest Neighbors
Support Vector Machine
XGBoost


Implementation of unsupervised learning algorithms:

K-Means Clustering
K-Means++ Clustering
DBSCAN Clustering


Comprehensive model comparison and visualization

2. Credit Card Fraud Detection (credit_card.py)
This script focuses specifically on credit card fraud detection:
Features:

Data preprocessing and standardization
Training and evaluation of various classification models:

Logistic Regression
Random Forest
Decision Tree
KNN
SVM
XGBoost


Performance visualization with confusion matrices and ROC curves
Unsupervised learning with K-Means and DBSCAN
Model comparison and evaluation

3. Intel Image Classification (cnn.py)
This script implements an image classification system using convolutional neural networks:
Features:

Loading and preprocessing of the Intel Image Dataset
Data augmentation for improved model robustness
Implementation of both CNN and MLP models
Hyperparameter tuning with grid search
Model evaluation with confusion matrices
Interactive training and visualization interface
Final model training with best parameters

Requirements
The code requires the following Python libraries:

numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras
xgboost
opencv-python
joblib

You can install the required packages using:
bashpip install -r requirements.txt
Usage
Fraud Analysis System
bashpython heuristics.py
Follow the interactive menu to load data, train models, and visualize results.
Credit Card Fraud Detection
bashpython credit_card.py
Make sure to have a "creditcard.csv" file in the same directory.
Intel Image Classification
bashpython cnn.py
Ensure you have the Intel Image Dataset in the specified directory.
Dataset Information

The fraud detection systems use a credit card transaction dataset with features like income, area, employment status, gender, balance, and a fraud indicator.
The image classification system uses the Intel Image Classification dataset with categories for buildings, forest, glacier, mountain, sea, and street images.

Results
The system generates various output files:

Cleaned data CSV
Data comparison visualizations
Correlation matrices
Model performance metrics
Confusion matrices
ROC curves
Trained model files (.pkl)
Clustering visualizations

intel image classification dataset has been removed from the zip folder for storage. 
please visit the following link to view the dataset https://www.kaggle.com/datasets/puneet6060/intel-image-classification


License
This project is for educational purposes. Please respect the licenses of the libraries and datasets used.
