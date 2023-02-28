import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Load data from CSV file
data = pd.read_csv('PositionSalaries.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# Create a decision tree regression model and fit the data
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Visualize the decision tree regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Salary vs Position Level (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predict the salary of a new employee using the model
new_position = [[6.5]]
new_salary = regressor.predict(new_position)
print("Predicted salary for position level 6.5 is $", int(new_salary))
