# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

print(tf.__version__)   # Make sure you have tensorflow version 2.0.0 or higher!

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Datasets/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values  # 1 == leave the bank | 0 == stay with the bank

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One Hot Encoding the "Geography" column
from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)     # Scale ALL!
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
ann = keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(keras.layers.Dense(units=6, activation='relu'))      # Pick any number they say(for units)...

# Adding the second hidden layer
ann.add(keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(keras.layers.Dense(units=1, activation='sigmoid'))     # if this is non-binary classification you will have to put more than: units>2 | activation='softmax' (WHEN PREDICTING MORE THAN 2 CATEGORIES!!!)

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])     # if this is non-binary classification you will have to put loss='categorical_crossentropy' (WHEN PREDICTING MORE THAN 2 CATEGORIES!!!)
                                                        # Can choose more than one metrics

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size=32, epochs=20)
# Can do it for the Test set too !

# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""



"""
Therefore, our ANN model predicts that this customer stays in the bank!
Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
"""

# Predicting the Test set results


# Making the Confusion Matrix

