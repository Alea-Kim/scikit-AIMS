import sys
import commands
import subprocess
import numpy as np
import cv2

# read each labels
test_label = np.loadtxt('./USPS_jpg/test_label.txt')
trai_label = np.loadtxt('./USPS_jpg/trai_label.txt')

# read test images and write a data file
arr = np.empty((0,256), int)
for i in range(0,4649):
    filename='./USPS_jpg/test_'+str(i)+'.jpg'
    test_img = cv2.imread(filename, 0)
    tmp = test_img.reshape(1,256)
    arr = np.append(arr, tmp, axis=0)
np.savetxt('test_data.txt', arr, fmt='%i')

# read training images and write a data file
arr = np.empty((0,256), int)
for i in range(0,4649):
    filename='./USPS_jpg/trai_'+str(i)+'.jpg'
    test_img = cv2.imread(filename, 0)
    tmp = test_img.reshape(1,256)
    arr = np.append(arr, tmp, axis=0)
np.savetxt('trai_data.txt', arr, fmt='%i')
