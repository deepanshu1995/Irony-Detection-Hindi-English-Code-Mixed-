from __future__ import division
import os
import numpy 
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split  # Random split into training and test dataset.
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
#from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from build_feature_vector import *
from format_data import *


id_tweet_map = create_id_tweet_map()
id_class_map = create_id_class_map()

X, Y = TrainingData(id_tweet_map, id_class_map)

# Convert list into a array
X = numpy.asarray(X)
Y = numpy.asarray(Y)
#print Y
#Y = MultiLabelBinarizer().fit_transform(Y)
#Y = LabelEncoder().fit_transform(Y)
#print Y.shape

X = SelectKBest(chi2, k=1400).fit_transform(X,Y)
#print X


kf = KFold(n_splits=10)
fold = 0

accuracy = 0
for train_idx, test_idx in kf.split(X):
		fold = fold + 1
		X_train, X_test = X[train_idx], X[test_idx]
		Y_train, Y_test = Y[train_idx], Y[test_idx]
		#print X_train
		#print Y_train
		#print(X_train.shape)
		#print(Y_train.shape)
		clf = svm.SVC(kernel = 'rbf', C=10)
		clf.fit(X_train, Y_train)
		predictions = clf.predict(X_test)
		score = accuracy_score(Y_test, predictions)
		print(classification_report(Y_test, predictions))
		accuracy = accuracy + score
		print("Score for fold %d: %.3f" %(fold, score))

print "Accuracy : " , round(accuracy/10, 3)

