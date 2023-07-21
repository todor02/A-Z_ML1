# Polynomial Regression
# Formula: y = b₀ + b₁x₁ + b₂x₁² + ... + bₙ x₁ⁿ

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('DataSets/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(x, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualising the Linear Regression results |JUST SEEING WHAT IS THE DIFFERENCE!|
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_1.predict(x), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
print('If we visualise the Linear Regressing model:')
print('We will see that its not optimal for this dataset!')

# Visualising the Polynomial Regression results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(x_poly), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_pred_1 = lin_reg_1.predict([[6.5]])
print('\nLinear Prediction for a salary between 6 and 7 (BAD PREDICTION):\n' + str(lin_pred_1))

# Predicting a new result with Polynomial Regression
lin_pred_2 = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print('\nPolynomial Prediction for a salary between 6 and 7:\n' + str(lin_pred_2))
