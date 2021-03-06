import numpy as np #/
from sklearn.datasets import load_iris #/
from sklearn import svm #/
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


iris = load_iris() #/
tmp = iris.data #/ iris['data']== iris.data?

# select two features
X = np.c_[tmp[:,1],tmp[:,3]]
feature_names = iris.feature_names #/ feature_names = data.feature_names?
y = iris.target #/

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

h = 0.02 #mesh size

clf = svm.SVC(gamma=0.001, kernel='linear')
y_pred = clf.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')
    print(cm)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
#np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=feature_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=feature_names, normalize=True,
                      title='Normalized confusion matrix')

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
