# Eclat

# Run the following command in the terminal to install the apyori package: pip install apyori
# !pip install apyori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('DataSets/Market_Basket_Optimisation.csv', header=None)
transactionsList = []
for i in range(0, 7501):
  transactionsList.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training the Eclat model on the dataset
from apyori import apriori
rules = apriori(transactions=transactionsList, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
# Recommended to keep all!

# Visualising the results:
# Displaying the first results coming directly from the output of the apriori function
results = list(rules)
print(results)

# Putting the results well organised into a Pandas DataFrame
def inspect(results):
  lhs         = [tuple(result[2][0][0])[0] for result in results]
  rhs         = [tuple(result[2][0][1])[0] for result in results]
  supports    = [result[1] for result in results]
  return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])

# Displaying the results sorted by descending support
print("\nSorted by descending support:\n")
print(resultsinDataFrame.nlargest(n=10, columns='Support'))
