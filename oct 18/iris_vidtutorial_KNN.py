from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

print X.shape
print y.shape

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)

x_new = [[3, 5, 4, 2],[5, 4, 3 , 2]]

mika = knn.predict(x_new)

print mika
