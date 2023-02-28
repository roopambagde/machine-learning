import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
data = pd.read_csv('ShopCustomers.csv')
X = data.iloc[:, [3, 4]].values
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute the linkage matrix using Ward's method
Z = linkage(X_scaled, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
plt.show()

# Plot the clusters on a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(X_scaled[:,0], X_scaled[:,1])
plt.title('Hierarchical Clustering Results')
plt.xlabel('Standardized Annual Income')
plt.ylabel('Standardized Spending Score')
plt.show()
