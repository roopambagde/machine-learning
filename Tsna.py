import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate some random high-dimensional data
data = np.random.randn(1000, 100)

# Reduce the dimensionality using t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
data_tsne = tsne.fit_transform(data)

# Plot the results
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], s=5)
plt.show()
