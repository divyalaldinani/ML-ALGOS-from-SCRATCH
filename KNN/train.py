import numpy as np # type: ignore
from sklearn import datasets # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import ListedColormap # type: ignore

from KNN import KNN

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])


# In summary, KNN can be effectively applied to clustered data for classification, provided that the data is 
# well-structured and the parameters are carefully chosen. 
data = datasets.load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


#plotting the data
plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s = 20)
plt.show()


clf = KNN(k = 7)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)