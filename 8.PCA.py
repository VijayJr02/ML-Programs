import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
df = pd.read_csv('pendigits_txt.csv').dropna()

# Split features and labels
X = StandardScaler().fit_transform(df.iloc[:, :-1])
y = df.iloc[:, -1]  # Extract target labels

# Apply PCA
X_pca = PCA(n_components=3).fit_transform(X)

# 3D Scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.6)

# Add colorbar
legend = fig.colorbar(scatter)
legend.set_label("Digit Label")

plt.show()
