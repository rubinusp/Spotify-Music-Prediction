import json
from sklearn.feature_extraction import DictVectorizer


class DataExtractor:

    def __init__(self):

        self.vec = DictVectorizer()
        self.data = None

    def load_json(self):
        """ It supports method chaining, e.g. dataExtractor().load_json().to_array()

        :return: self
        """

        f = open('./dataset/songs.json', )

        self.data = json.load(f)

        f.close()

        return self

    def to_array(self):

        print(self.data)
        self.data = self.vec.fit_transform(self.data).toarray()
        print(self.data)

        return self.data


extr = DataExtractor()
arr = extr.load_json().to_array()