import numpy as np
from sklearn.datasets import load_iris
from sklearn import neighbors
from matplotlib import pyplot as plt

data = load_iris()
tmp = data['data']
# select two features
X = np.c_[tmp[:,1],tmp[:,3]]
feature_names = data['feature_names']
del feature_names[0]
del feature_names[1]
label = data['target']

h = .02 # step size in the mesh
knn=neighbors.KNeighborsClassifier(n_neighbors=1)
# we create an instance of Neighbours Classifier and fit the data.
knn.fit(X, label)

plt.figure(1)

# Plot the decision boundary. For that, we will asign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.2)

# plot samples
for t, marker, c in zip(list(range(3)), ">ox", "rgb"):
	plt.scatter(X[label == t, 0], X[label == t, 1], marker=marker, c=c)

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.xticks()
plt.yticks()
plt.show()
