import numpy as np #/
from sklearn import neighbors #/
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

print "Handwritten Digits\nwith K-Nearest Neighbor(k=1)"
##--- load USPS dataset ---
##--- test samples and their labels ---
X_test = np.loadtxt('test_data.txt')
# print X_test.shape
X_test = X_test.astype(np.float32) # original data
test_label = np.loadtxt('./USPS_jpg/test_label.txt')
test_label = test_label.flatten().astype(np.int32) # test sample's label
##--- training samples and their labels ---
X_trai = np.loadtxt('trai_data.txt')
X_trai = X_trai.astype(np.float32) # original data
trai_label = np.loadtxt('./USPS_jpg/trai_label.txt')
trai_label = trai_label.flatten().astype(np.int32) # training sample's label

# preprocessing
X_trai /= 255.0
X_test /= 255.0

h = 0.02 #/ DEAFULT? :O step size in the mesh

knn=neighbors.KNeighborsClassifier(n_neighbors=1) #/ K = 1 okay we made an instance that can do knn
y_pred = knn.fit(X_trai, trai_label).predict(X_test)


def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix'):
    print('Confusion matrix')
    print(cm)

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, y_pred)
#np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix)