import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC

# Load the data
data = pd.read_csv('SocialNetwork_Ads.csv')

# Split the data into features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Define the SVM model
model = SVC()

# Define the number of folds for cross-validation
num_folds = 5

# Define the evaluation metric
scoring = 'accuracy'

# Define the K-fold cross-validation object
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform K-fold cross-validation
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)

# Print the results
print("Results: ", results)
print("Mean accuracy: %.2f%%" % (results.mean()*100))
print("Standard deviation: %.2f%%" % (results.std()*100))
