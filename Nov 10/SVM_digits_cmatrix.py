import numpy as np #/
from matplotlib import pyplot as plt
from sklearn import svm, metrics #/
from sklearn.metrics import confusion_matrix

print "Handwritten Digits\nwith Support Vector Machine (Linear)"
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

svm = svm.SVC(gamma=0.001, kernel='linear')
y_pred = svm.fit(X_trai, trai_label).predict(X_test)


def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix'):
    print('Confusion matrix')
    print(cm)

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, y_pred)
#np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix)

print("Classification report for classifier %s:\n%s\n"
      % (svm, metrics.classification_report(test_label, y_pred)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_label, y_pred))
