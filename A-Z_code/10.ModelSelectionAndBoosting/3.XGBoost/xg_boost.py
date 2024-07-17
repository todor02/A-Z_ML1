# XGBoost

# NOTE: XGBClassifier is used for classification & XGBRegressor is used for regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Importing the dataset
dataset = pd.read_csv('Datasets/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode labels !!!
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)    # y_encoded instead of y!!!

# Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# [[85  2]
#  [ 1 49]]
print(accuracy_score(y_test, y_pred))
# 0.9781021897810219

# Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(accuracies)
# [0.97826087 0.93478261 0.97826087 0.91304348 0.93478261 0.95652174
# 0.93478261 0.93478261 0.97826087 0.93478261]
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100)) # Accuracy: 96.53 %
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100)) # Standard Deviation: 2.63 %
