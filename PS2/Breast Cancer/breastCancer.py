#from http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data

import pandas as pd
from sklearn.utils import shuffle

np.random.seed(42)

columns = ['ID','1', '2', '3', '4', '5', '6', '7', '8','9','class']
data = pd.read_csv('breast-cancer-wisconsin.data', names = columns)

del data['ID']
data = data[~data.isin(['?'])]

data = data.dropna()

data = shuffle(data)

labels = data['class'].copy()
del data['class']

labels[labels == 2] = 0
labels[labels == 4] = 1

labels.to_csv('breastCancerLabels.csv', header = None, index = False)
data.to_csv('breastCancerData.csv', header = None, index = False)

