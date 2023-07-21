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
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualising the Linear Regression results


# Visualising the Polynomial Regression results


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)


# Predicting a new result with Linear Regression


# Predicting a new result with Polynomial Regression
