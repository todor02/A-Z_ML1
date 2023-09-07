# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('DataSets/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values   # This will only take 2 columns because we need to visualise it in 2D plot
