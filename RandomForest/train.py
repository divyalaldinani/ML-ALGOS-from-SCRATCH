from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
from RandomForest import RandomForest


data = datasets.load_digits() #multiple digit classfication from 0 to 9, can try other datasets also -> breast cancer, diabetes and iris
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)
print(X.shape)
classifier = RandomForest(n_trees = 20)
classifier.fit(X_train, y_train)


def accuracy(y_test, predictions):
    return np.sum(y_test == predictions)/len(y_test)


predictions = classifier.predict(X_test)
accuracy_ = accuracy(y_test, predictions)
print(accuracy_)