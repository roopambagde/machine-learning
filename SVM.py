import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
dataset = pd.read_csv('SocialNetwork_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
fig, ax = plt.subplots()
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
ax.set_xlabel('Age')
ax.set_ylabel('Estimated Salary')
ax.set_title('SVM Classification Results')
plt.show()

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print(cm)
