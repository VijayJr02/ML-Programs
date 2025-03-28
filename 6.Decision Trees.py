import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
import graphviz

# Load dataset
data = pd.read_csv('wine_quality.csv')

# Check and remove duplicates
data = data.drop_duplicates()

# Split features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Ensure class balance in train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Train Decision Tree with constraints to prevent overfitting
clf = DecisionTreeClassifier(max_depth=4, min_samples_split=2, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Visualize decision tree
dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=y.unique().astype(str),
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("/content/decision_tree")
