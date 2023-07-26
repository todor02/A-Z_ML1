# Logistic Regression
# Algorithm: In(p / 1 - p) = b₀ + b₁x₁ + ... + bₙ xₙ ||| p is probability
# Likelihood: multiply each point p ||| The higher number the better Logistic Regression curve

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('DataSets/Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature Scaling

# Training the Logistic Regression model on the Training set

# Predicting a new result

# Predicting the Test set results

# Making the Confusion Matrix

# Visualising the Training set results

# Visualising the Test set results
