import numpy as np #/
from matplotlib import pyplot as plt
from sklearn import svm, metrics #/
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix'):
    print('Confusion matrix')
    print(cm)

def diagonal_sum(cnf, n):
    SUM = 0
    for i in range(0,n):
        for j in range(0,n):
            if i == j:
                SUM = SUM + cnf[i,j]
    return SUM

def get_acc(total, add):
    total = float(total)
    add = float(add)
    return add/total

print "Handwritten Digits\nwith Support Vector Machine"
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
total_test = 4649
number = 10

#np.set_printoptions(precision=2)
#plot_confusion_matrix(cnf_matrix)

x_axis = []
to_plot = []

for i in range(1,number+1): #1 and 2 == so to plot is 2
    svm1 = svm.SVC(gamma=(0.01*i))
    y_pred = svm1.fit(X_trai, trai_label).predict(X_test)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_label, y_pred)    #np.set_printoptions(precision=2)
    #plot_confusion_matrix(cnf_matrix)

    a = get_acc(total_test, diagonal_sum(cnf_matrix, 10))
    to_plot.append(a)

print to_plot

for i in range (1,number+1):
    x_axis.append(i)

plt.plot(x_axis, to_plot)
plt.axis([1,number+1,0.8,1])
plt.show()


#print("Classification report for classifier %s:\n%s\n"
#      % (svm, metrics.classification_report(test_label, y_pred)))
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_label, y_pred))
