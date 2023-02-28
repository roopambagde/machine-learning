import numpy as np
from sklearn.decomposition import FastICA

# Generate a random signal with 3 independent sources
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)
s1 = np.sin(2 * time)
s2 = np.sign(np.sin(3 * time))
s3 = np.random.randn(n_samples)
S = np.c_[s1, s2, s3]

# Mix the sources using a random mixing matrix
A = np.array([[0.5, 1.0, 0.0], [1.5, 0.5, 0.5], [0.0, 2.0, 1.0]])
X = np.dot(S, A.T)

# Perform ICA on the mixed signal
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)

# Plot the results
import matplotlib.pyplot as plt

plt.figure()

models = [X, S, S_]
names = ['Mixed Signal', 'True Sources', 'ICA Estimated Sources']

for i, (model, name) in enumerate(zip(models, names)):
    plt.subplot(3, 1, i + 1)
    plt.title(name)
    for sig in model.T:
        plt.plot(sig)

plt.tight_layout()
plt.show()
