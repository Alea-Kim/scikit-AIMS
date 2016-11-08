import numpy as np #/
from sklearn.datasets import load_iris #/
from sklearn import svm #/
from matplotlib import pyplot as plt

iris = load_iris() #/
tmp = iris.data #/ iris['data']== iris.data?

# select two features
X = np.c_[tmp[:,1],tmp[:,3]]
feature_names = iris.feature_names #/ feature_names = data.feature_names?

#del feature_names[0] #Why delete? :O
#del feature_names[1] #Why delete? :O

y = iris.target #/

h = 0.02 #mesh size

clf = svm.SVC(gamma=0.001)
clf.fit(X, y)

plt.figure(1) #/ FIRST FIGURE

x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.2)


#print("Classification report for classifier %s:\n%s\n"
#      % (clf, metrics.classification_report(y, Z)))
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(y, Z))


# plot samples #gets
for t, shape, color in zip(list(range(3)), ">ox", "rbg"):
    plt.scatter(X[y == t, 0], X[y == t, 1], marker=shape, c=color)


plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])

plt.xticks()
plt.yticks()
plt.show()
