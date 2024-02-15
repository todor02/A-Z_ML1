# Artificial Neural Network

# Importing the libraries


# Part 1 - Data Preprocessing

# Importing the dataset


# Encoding categorical data
# Label Encoding the "Gender" column


# One Hot Encoding the "Geography" column


# Splitting the dataset into the Training set and Test set


# Feature Scaling


# Part 2 - Building the ANN

# Initializing the ANN


# Adding the input layer and the first hidden layer


# Adding the second hidden layer


# Adding the output layer


# Part 3 - Training the ANN

# Compiling the ANN


# Training the ANN on the Training set


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

