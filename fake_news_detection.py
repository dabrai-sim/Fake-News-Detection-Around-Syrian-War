import numpy as np
import pandas as pd
import nltk
import warnings
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, jaccard_score, 
    precision_score, recall_score, roc_curve, roc_auc_score
)
import chardet

warnings.filterwarnings('ignore')

# Import dataset
with open('data/fakenewsDataset.csv', 'rb') as file:
    result = chardet.detect(file.read())
encoding = result['encoding']

df = pd.read_csv('data/fakenewsDataset.csv', encoding=encoding)

# Preprocessing
df.drop(['Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'unit_id', 'date', 'location'], axis=1, inplace=True)

# Label conversion
df.labels.replace([0, 1], ['Fake', 'Real'], inplace=True)

# Combine article content and title
x = (df['article_content'] + ' ' + df['article_title']).to_numpy()
y = df['labels'].to_numpy()

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Text vectorization
tfidf = TfidfVectorizer()
x_train_vect = tfidf.fit_transform(x_train)
x_test_vect = tfidf.transform(x_test)

def plot_roc_curve(y_test, prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, prob, pos_label='Real')
    roc_auc = roc_auc_score(y_test, prob)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_vect, y_train)
knn_pred = knn.predict(x_test_vect)
print("KNN Results:")
print("Accuracy: ", accuracy_score(y_test, knn_pred))
print("F1 score: ", f1_score(y_test, knn_pred, pos_label='Real'))
print("Jaccard's index: ", jaccard_score(y_test, knn_pred, pos_label='Real'))
print("Precision: ", precision_score(y_test, knn_pred, pos_label='Real'))
print("Recall: ", recall_score(y_test, knn_pred, pos_label='Real'))

knn_prob = knn.predict_proba(x_test_vect)[:, 1]
plot_roc_curve(y_test, knn_prob, "KNN")

# Naive Bayes Model
nb = MultinomialNB()
nb.fit(x_train_vect, y_train)
nb_pred = nb.predict(x_test_vect)
print("\nNaive Bayes Results:")
print("Accuracy: ", accuracy_score(y_test, nb_pred))
print("F1 score: ", f1_score(y_test, nb_pred, pos_label='Real'))
print("Jaccard's index: ", jaccard_score(y_test, nb_pred, pos_label='Real'))
print("Precision: ", precision_score(y_test, nb_pred, pos_label='Real'))
print("Recall: ", recall_score(y_test, nb_pred, pos_label='Real'))

nb_prob = nb.predict_proba(x_test_vect)[:, 1]
plot_roc_curve(y_test, nb_prob, "Naive Bayes")

# Logistic Regression Model
lr = LogisticRegression()
lr.fit(x_train_vect, y_train)
lr_pred = lr.predict(x_test_vect)
print("\nLogistic Regression Results:")
print("Accuracy: ", accuracy_score(y_test, lr_pred))
print("F1 score: ", f1_score(y_test, lr_pred, pos_label='Real'))
print("Jaccard's index: ", jaccard_score(y_test, lr_pred, pos_label='Real'))
print("Precision: ", precision_score(y_test, lr_pred, pos_label='Real'))
print("Recall: ", recall_score(y_test, lr_pred, pos_label='Real'))

lr_prob = lr.predict_proba(x_test_vect)[:, 1]
plot_roc_curve(y_test, lr_prob, "Logistic Regression")
