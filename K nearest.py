import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
social_network_data = pd.read_csv('SocialNetwork_Ads.csv')

# split data into training and testing sets
X = social_network_data.iloc[:, :-1].values
y = social_network_data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train KNN classifier with k=5
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# make predictions on test data
y_pred = classifier.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# visualize the results
fig, ax = plt.subplots()
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
ax.set_xlabel('Age')
ax.set_ylabel('Estimated Salary')
ax.set_title('KNN Classification Results')
plt.show()
