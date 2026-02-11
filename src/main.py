# Wine Quality Prediction Project
# Author: K Tarun Kumar

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("data/winequality-red.csv")

# Convert to binary classification (Good = 1, Bad = 0)
data['quality_label'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Features & Target
X = data.drop(['quality', 'quality_label'], axis=1)
y = data['quality_label']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Logistic Regression ----------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, log_pred)
print("Logistic Regression Accuracy:", log_accuracy)

# ---------------- SVM ----------------
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)
print("SVM Accuracy:", svm_accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, svm_pred)
sns.heatmap(cm, annot=True)
plt.title("SVM Confusion Matrix")
plt.show()
