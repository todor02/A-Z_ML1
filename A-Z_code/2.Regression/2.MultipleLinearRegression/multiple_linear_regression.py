# Multiple Linear Regression
# Formula: ŷ = b₀ + b₁x₁ + b₂x₂ + ... + bₙ xₙ

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('DataSets/50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print("Default 'x' dataset")
print(x)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")
x = np.array(ct.fit_transform(x))
print("\nEncoded independent variable 'x' dataset")
print(x)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the Test set result
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print("\nPredicting the Test set result ||| The closer the better !")
print("Predicted --- Actual")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

# NOTE TO SELF ::: Backward Elimination on Section:7 71.
