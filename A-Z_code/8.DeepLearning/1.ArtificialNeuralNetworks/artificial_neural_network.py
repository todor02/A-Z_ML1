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
Use the ANN model to predict if the customer with the following informations will leave the bank: 
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
"""
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
# =0.036 chance to leave the bank => 3,6%

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print("Predicted --- Actual")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))   # ~0.84
