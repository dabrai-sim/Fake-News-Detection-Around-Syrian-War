# Fake News Detection Around Syrian War

## Project Overview
This project aims to build machine learning models to detect fake news related to the Syrian War using K-Nearest Neighbours (KNN), Naïve Bayes, and Logistic Regression techniques. The dataset is obtained from [Zenodo](https://zenodo.org/record/2607278).

## Features
- Data preprocessing and text vectorization using TF-IDF.
- Implementation of KNN, Naïve Bayes, and Logistic Regression models.
- Evaluation of models using various metrics:
  - Accuracy
  - F1-score
  - Jaccard’s index
  - Precision & Recall
  - AUC-ROC curve

## Technical Stack
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- NLTK

## File Structure
- `fake_news_detection.py`: Main script for training and evaluating models.
- `requirements.txt`: Required Python packages.
- `data/fakenewsDataset.csv`: Fake news dataset.

## Data Source
The dataset can be downloaded from [Zenodo](https://zenodo.org/record/2607278).
