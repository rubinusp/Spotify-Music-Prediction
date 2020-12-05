import json
import numpy as np
from copy import deepcopy
from sklearn.feature_extraction import DictVectorizer


class DataExtractor:

    def __init__(self):

        self.vec = DictVectorizer()
        self.data = None

    def load_json(self):
        """ It supports method chaining, e.g. dataExtractor().load_json().to_array()

        :return: self
        """
        f = open('dataset/combinedData.json', 'r')
        self.data = json.load(f)
        f.close()

        return self

    def to_array(self):

        self.data = self.vec.fit_transform(self.data).toarray()
        self.data = np.unique(self.data, axis=0)
        self.data[:, [10, -1]] = self.data[:, [-1, 10]]
        self.data[:, [10, 11]] = self.data[:, [11, 10]]
        self.data[:, [11, 12]] = self.data[:, [12, 11]]
        self.data[:, [13, 12]] = self.data[:, [12, 13]]
        return self.data

    def normalize(self, X):
        result = deepcopy(X)
        for j in range(result.shape[1]):
            xmax = np.amax(result[:, j])
            xmin = np.amin(result[:, j])
            result[:, j] = (result[:, j] - xmin) / (xmax - xmin)
        return result

##extr = DataExtractor()
#arr = extr.load_json().to_array()
