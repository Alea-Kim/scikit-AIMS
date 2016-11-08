#coding: utf-8
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

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

class UspsModel(chainer.Chain):
    def __init__(self):
        super(UspsModel, self).__init__(
                l1=L.Linear(256, 100),
                l2=L.Linear(100, 10)
        )

    def __call__(self, x, t, train):

        x = chainer.Variable(x)
        t = chainer.Variable(t)

        h = F.relu(self.l1(x))
        h = self.l2(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.accuracy(h, t)

# for outputing a class label
def predictedlabel(x_data):
    x = chainer.Variable(x_data)
    h = F.relu(model.l1(x))
    h = model.l2(h)
    return h.data.argmax()

model = UspsModel()
optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(100):
    model.zerograds()
    loss, acc = model(X_trai, trai_label, train=True)
    loss.backward()
    optimizer.update()
    print epoch, "training accuracy ", acc.data

acc = model(X_test, test_label, train=False)
print "test accuracy ", acc.data

# performance evaluation with confusion matrix
conf = np.zeros([10,10])
for ii in range(4649):
    x = np.array([X_test[ii]])
    y = predictedlabel(x)
    t = test_label[ii]
    conf[t][y]+=1
print conf
