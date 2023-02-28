import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data into a pandas DataFrame
df = pd.read_csv("Wine.csv")

# Separate the features and target variable
X = df.drop('Customer_Segment', axis=1)
y = df['Customer_Segment']

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create a PCA object
pca = PCA(n_components=2)

# Fit the PCA object to the standardized data
X_pca = pca.fit_transform(X_std)

# Plot the results
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
markers = ['o', 's', 'D']
for target, color, marker in zip(np.unique(y), colors, markers):
    plt.scatter(X_pca[y == target, 0],
                X_pca[y == target, 1],
                c=color, marker=marker,
                label=target)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
