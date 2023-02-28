import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load data from CSV file
data = pd.read_csv('PositionSalaries.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# Feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Fit the SVR model to the dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, y)
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Salary vs Position Level (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
new_position = np.array([[6.5]])
new_position = sc_X.transform(new_position)
new_salary = sc_y.inverse_transform(regressor.predict(new_position))
print("Predicted salary for position level 6.5 is $", int(new_salary))
