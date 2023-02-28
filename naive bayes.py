import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
dataset = pd.read_csv('SocialNetwork_Ads.csv')
# Extract the features and labels from the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Use the classifier to predict the labels of the test data
y_pred = classifier.predict(X_test)

# Visualize the classification results
fig, ax = plt.subplots()
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
ax.set_xlabel('Age')
ax.set_ylabel('Estimated Salary')
ax.set_title('Naive Bayes Classification Results')
plt.show()

# Evaluate the performance of the classifier using a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
