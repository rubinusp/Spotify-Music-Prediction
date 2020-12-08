#Basic classifier

import dataextractor as de
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

extr = de.DataExtractor()
data = extr.load_json().to_array()

labels = data[:,-1]
data = data[:,:-1]
data = extr.normalize(data)

per = .7
new_labels = extr.catagorize(labels, per)
X_train, X_test, y_train, y_test = train_test_split(data, new_labels, stratify=new_labels)
clf = MLPClassifier(nesterovs_momentum=True, max_iter=500).fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)