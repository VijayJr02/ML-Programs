import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import seaborn as sns

data = sns.load_dataset('titanic').dropna(subset=['age', 'fare', 'sex', 'class', 'survived'])
data = pd.get_dummies(data, columns=['sex', 'class', 'embarked'], drop_first=True)

X, y = data[['age', 'fare', 'sex_male']], data['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
y_probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.legend()
plt.show()
