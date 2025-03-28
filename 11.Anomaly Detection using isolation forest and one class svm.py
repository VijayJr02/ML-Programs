import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('creditcard.csv')

# Separate features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    'Isolation Forest': IsolationForest(contamination=0.0017, random_state=42),
    'One-Class SVM': OneClassSVM(nu=0.0017, kernel='rbf', gamma='auto')
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train)
    preds = [1 if x == -1 else 0 for x in model.predict(X_test)]
    print(f"{name} Accuracy: {accuracy_score(y_test, preds):.4f}")
