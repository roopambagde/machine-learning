import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("Data.csv")

# Split into features and target variable
X = data.drop("Country", axis=1)
y = data["Country"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes and types of X_train and y_train
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Type of X_train:", type(X_train))
print("Type of y_train:", type(y_train))

# Convert the data to XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train.values.ravel())
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the XGBoost hyperparameters
params = {
    "max_depth": 3,
    "eta": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "error",
    "seed": 42
}

# Train the model
num_rounds = 50
xgb_model = xgb.train(params, dtrain, num_rounds)

# Make predictions on the test set
y_pred = xgb_model.predict(dtest)

# Convert predicted probabilities to binary predictions
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)
