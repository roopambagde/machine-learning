import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

# Generate sample data
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.05)

# Apply Kernel PCA with rbf kernel
kpca = KernelPCA(kernel='rbf', gamma=15, n_components=2)
X_kpca = kpca.fit_transform(X)

# Plot results
plt.figure()
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Kernel PCA with RBF kernel')
plt.show()
