import numpy as np
from sklearn.datasets import load_iris
from sklearn import neighbors
from matplotlib import pyplot as plt

import chainer
import chainer.functions  as F
import chainer.links as L
from chainer import training, datasets
from chainer.training import extensions

import sys
from chainer import Variable

data = load_iris()
tmp = data['data']
# select two features
X = np.c_[tmp[:,1],tmp[:,3]]
X = X.astype(np.float32)
feature_names = data['feature_names']
del feature_names[0]
del feature_names[1]
label = data['target']
label = label.flatten().astype(np.int32)

train ,test= datasets.split_dataset_random(chainer.datasets.TupleDataset(X,label),75)
train_iter = chainer.iterators.SerialIterator(train, 10)
test_iter = chainer.iterators.SerialIterator(test, 1,repeat=False, shuffle=False)

d=2
nUnits = 64
nC=3
nepochs=40

class IrisModel(chainer.Chain):
    def __init__(self):
        super(IrisModel,self).__init__(
                l1 = L.Linear(d,nUnits),
                l2 = L.Linear(nUnits,nC))
    def __call__(self,x):
         h = F.sigmoid(self.l1(x))
         return self.l2(h)

model = L.Classifier(IrisModel())
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (nepochs, 'epoch'), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()

# performance evaluation with confusion matrix
conf = np.zeros([nC,nC])
for ii in range(75):
    x = np.array([test[ii][0]])
    h = model.predictor(x)
    y = F.softmax(h).data.argmax(axis=1)
    t = test[ii][1]
    conf[t][y]+=1
print conf

# Plot the decision boundary. For that, we will asign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
plt.figure(1)
h = .02
x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
M=np.c_[xx.ravel(), yy.ravel()]
Z=np.zeros(len(M))
for jj in xrange(len(M)-1):
    x = Variable(M[jj].reshape(1,d).astype(np.float32))
    h = model.predictor(x)
    y = F.softmax(h).data.argmax(axis=1)
    Z[jj]=y

Z = Z.reshape(xx.shape)
# 1-> the number of contour lines is one
plt.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.2)

# plot samples
for t, marker, c in zip(list(range(3)), ">ox", "rgb"):
    plt.scatter(X[label == t, 0], X[label == t, 1], marker=marker, c=c)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.xticks()
plt.yticks()
plt.show()
