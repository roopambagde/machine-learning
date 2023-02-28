import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# Load data from CSV file
data = pd.read_csv('PositionSalaries.csv')
# Separate input variable (X) and output variable (y)
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# Fit polynomial regression to the dataset
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Visualize the polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Position vs Salary (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# Predict the salary of a new employee using the model
new_position = [[6.5]]
new_salary = lin_reg.predict(poly_reg.fit_transform(new_position))
print("Predicted salary for position level 6.5 is $", int(new_salary))
