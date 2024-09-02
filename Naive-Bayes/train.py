from sklearn import datasets # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import numpy as np # type: ignore
from NaiveBayes import SimpleNbClassifier # type: ignore

data = datasets.load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.2, random_state=None
)

classifier = SimpleNbClassifier()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

def accuracy(y_test, predictions):
    return np.sum(y_test == predictions)/len(y_test)
#accuracy = number of sample whose label is correctly predicted/total number of samples

accuracy_of_Tree = accuracy(y_test, predictions )
print(accuracy_of_Tree)
