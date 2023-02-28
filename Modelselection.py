import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Load the data
data = pd.read_csv('SocialNetwork_Ads.csv')

# Split the data into features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a base estimator
estimator = DecisionTreeClassifier(max_depth=1)

# Initialize the AdaBoostClassifier
boosted_model = AdaBoostClassifier(base_estimator=estimator, n_estimators=50, learning_rate=1.0, random_state=42)
# Train the boosted model
boosted_model.fit(X_train, y_train)
# Evaluate the model on the test set
y_pred = boosted_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Tune the hyperparameters of the AdaBoostClassifier using grid search
from sklearn.model_selection import GridSearchCV
param_grid = {'estimator__max_depth': [1, 2, 4, 6], 'n_estimators': [10, 50, 100], 'learning_rate': [0.1, 0.5, 1.0]}
grid_search = GridSearchCV(boosted_model, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)
boosted_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
# Evaluate the final model on the test set
y_pred = boosted_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
