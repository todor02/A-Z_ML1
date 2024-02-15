# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Datasets/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)  # quoting=3 removes all quotes ( "" )

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')  # these are non-relevant words like "the", "a" ...
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  # This will take only the root of the word: "loved" => "love"
corpus = []     # new array for the cleaned dataset text, with we will later fill up
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])   # ^ == not || all non-letters are replaced by "space"
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    # if the "word" is not a stopword it will be included in the stemming =>
    all_stopwords = stopwords.words('English')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)     # First check with print(len(X[0])) then cut it down a little bit (because of random words)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("Predicted --- Actual")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))   # ~0.72
