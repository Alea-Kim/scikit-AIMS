#coding: utf-8
import numpy as np
import chainer
import chainer.functions as F
from chainer import optimizers
import chainer.links as L
from chainer import training
from chainer.training import extensions
import pylab
import matplotlib.pyplot as plt

##--- load USPS dataset ---
##--- training samples and their labels ---
X_trai = np.loadtxt('trai_data.txt')
X_trai = X_trai.reshape((len(X_trai), 1, 16, 16))
X_trai = X_trai.astype(np.float32) # original data
trai_label = np.loadtxt('./USPS_jpg/trai_label.txt')
trai_label = trai_label.flatten().astype(np.int32) # training sample's label

##--- test samples and their labels ---
X_test = np.loadtxt('test_data.txt')
# print X_test.shape
X_test = X_test.reshape((len(X_test), 1, 16, 16))
X_test = X_test.astype(np.float32) # original data
test_label = np.loadtxt('./USPS_jpg/test_label.txt')
test_label = test_label.flatten().astype(np.int32) # test sample's label

# preprocessing
X_trai /= 255.0
X_test /= 255.0

trai = chainer.datasets.TupleDataset(X_trai,trai_label)
test = chainer.datasets.TupleDataset(X_test,test_label)
#plt.imshow(X_trai[1][0], cmap=pylab.cm.gray_r, interpolation='nearest')
#plt.show()

# Network definition
class CNN(chainer.Chain):
    def __init__(self, train=True):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 20, 3, pad=1),
            conv2=L.Convolution2D(20, 50, 3, pad=1),
            l1=L.Linear(800, 100),
            l2=F.Linear(100, 10),
        )
        self.train = train

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.l1(h))
        return self.l2(h)

# Set up a neural network to train
model = L.Classifier(CNN())
# Setup an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
train_iter = chainer.iterators.SerialIterator(trai, batch_size=100)
test_iter = chainer.iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

# Set up a trainer
# For use of this program on CPU environment, change num of parameter "device" to be -1 from 0(zero)
updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (100, 'epoch'), out='result')
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
# Run the training
trainer.run()

# performance evaluation with confusion matrix
conf = np.zeros([10,10])
for ii in range(4649):
    x = np.array([test[ii][0]])
    h = model.predictor(x)
    y = F.softmax(h).data.argmax(axis=1)
    t = test[ii][1]
    conf[t][y]+=1
print conf
