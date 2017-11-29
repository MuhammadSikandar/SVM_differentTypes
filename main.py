import numpy as np
import pandas as pd
import sys
import sklearn
from sklearn import svm

from sklearn import metrics
from sklearn.metrics import accuracy_score
from time import time

from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

#print len(features_train)
# lets take the 10 percent of data
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

#print len(features_train)

# i will apply different varient of SVM

# 1) Simple SVM: it turns out to be "rbf" with C = 1 and gamma = auto, accuracy 61.6% OVR
#training
t0 = time()
clf = svm.SVC()
print clf
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
#print features_test[10,:]
#testing
t0 = time()
y_expect = labels_test
y_pred = clf.predict(features_test)
print sum(y_pred)
print accuracy_score(y_pred,y_expect)
print "testing time:", round(time()-t0, 3), "s"


# 2) Simple SVM: it turns out to be "rbf" with C = 1 and gamma = auto, accuracy 61.6% OVO
#training
t0 = time()
clf = svm.SVC(decision_function_shape='ovo')
print clf
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
#print features_test[10,:]
#testing
t0 = time()
y_expect = labels_test
y_pred = clf.predict(features_test)
print sum(y_pred)
print accuracy_score(y_pred,y_expect)
print "testing time:", round(time()-t0, 3), "s"




# 3) Linear SVM: OVR accuracy 89.8% C = 1
t0 = time()
clf = svm.LinearSVC()
print clf
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
#print features_test[10,:]
#testing
t0 = time()
y_expect = labels_test
y_pred = clf.predict(features_test)
print sum(y_pred)
print accuracy_score(y_pred,y_expect)
print "testing time:", round(time()-t0, 3), "s"

# 4) RBF SVM: OVR playing with C and gamma, by default C = 1 and gamma = 'auto'
# C = 1, gamma = 1 acc = 89.8%
# C = 1, gamma = 10 acc = 56.76%
# C = 1, gamma = 100 acc = 52%
# C = 10, gamma = 1 acc = 90.1%
# C = 100, gamma = 1 acc = 90.1%
# C = 1000, gamma = 1 acc = 90.1%
# gamma =
t0 = time()
clf = svm.SVC(kernel='rbf',C= 1000, gamma = 1)
print clf
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
#print features_test[10,:]
#testing
t0 = time()
y_expect = labels_test
y_pred = clf.predict(features_test)
print sum(y_pred)
print accuracy_score(y_pred,y_expect)
print "testing time:", round(time()-t0, 3), "s"

# 5) poly SVM: OVR acc
t0 = time()
clf = svm.SVC(kernel='poly', C= 1, gamma = 1)
print clf
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
#print features_test[10,:]
#testing
t0 = time()
y_expect = labels_test
y_pred = clf.predict(features_test)
print sum(y_pred)
print accuracy_score(y_pred,y_expect)
print "testing time:", round(time()-t0, 3), "s"