import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # Example dataset

# Load dataset (Replace this with your actual dataset)
data = load_iris()
X = data.data
y = data.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution
class_counts = pd.Series(y_train).value_counts()
print("Class Distribution in y_train:\n", class_counts)

# Determine the maximum possible value for cv (smallest class size)
min_samples_per_class = class_counts.min()
cv_value = min(5, min_samples_per_class)  # Reduce cv if necessary

# Define k-values for KNN
k_values = range(1, 11)  # Example k values from 1 to 10

# Perform cross-validation
cv_scores = [
    cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=cv_value).mean()
    for k in k_values
]

# Print results
for k, score in zip(k_values, cv_scores):
    print(f"K={k}, Cross-Validation Score: {score:.4f}")
