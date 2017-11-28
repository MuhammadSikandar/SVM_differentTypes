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
