from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np #/

y_true = [2,0,2,2,0,1]
y_pred = [0,0,2,2,0,2]
cnf = confusion_matrix(y_true, y_pred)

print cnf
SUM = 0

for i in range(0,3):
    for j in range(0,3):
        if i == j:
            SUM = SUM + cnf[i,j]

print SUM
