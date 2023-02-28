from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize the base classifier
base_classifier = DecisionTreeClassifier(max_depth=1)

# Initialize the AdaBoost classifier
adaboost_classifier = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=50)

# Fit the AdaBoost classifier to the training data
adaboost_classifier.fit(X_train, y_train)

# Predict the labels of the test data using the trained classifier
y_pred = adaboost_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
