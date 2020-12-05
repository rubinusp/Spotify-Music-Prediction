import dataextractor as de
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

extr = de.DataExtractor()
data = extr.load_json().to_array()

print(len(data))
print(len(data[0]))

# Acousticiness [0]
# Dancibility [1]
# Duration [2]
# Energy [3]
# Explicit [4]
# Instrumentalness [5]
# Key [6]
# Liveness [7]
# Loudness [8]
# Mode [9]
#d
# Speechiness [11]
# Tempo [12]
# Time signature [13]
# Valence [14]

print(data[0])
labels = data[:,-1]
data = data[:,:-1]
data = extr.normalize(data)
print(data[0])
print(labels[0])

X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=1, test_size=.3)
print(len(X_train))
print(len(X_test))

clf = MLPRegressor().fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                    random_state=1)
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)