from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from string import punctuation

import numpy as np
import sys


def tf_idf_matrix(dataset):
    vectorizer = CountVectorizer(min_df=3, analyzer=build_analyzer(), stop_words='english')
    transformer = TfidfTransformer()
    counts_data = vectorizer.fit_transform(dataset)
    matrix = transformer.fit_transform(counts_data)
    return matrix


def build_analyzer():
    analyzer = CountVectorizer().build_analyzer()
    stop_words_en = stopwords.words('english')
    combined_stopwords = set.union(set(stop_words_en), set(ENGLISH_STOP_WORDS), set(punctuation))

    def decorated_analyzer(doc):
        return [word for word in analyzer(doc) if word not in combined_stopwords and not word.isdigit()]

    return decorated_analyzer


class SLTransformer(object):
    def __init__(self, scaling, logarithm):
        self.use_scaling = scaling
        self.use_logarithm = logarithm
        self.epsilon = sys.float_info.epsilon

    def scaling(self, data_matrix):
        col_number = data_matrix.shape[1]
        for i in range(col_number):
            feature = data_matrix[:, i]
            average = feature.mean()
            zero_mean_feature = feature - average
            std_variance = np.linalg.norm(zero_mean_feature)
            data_matrix[:, i] = feature / std_variance
        return data_matrix

    def logarithm(self, data_matrix):
        return np.log(data_matrix + self.epsilon)

    def transform(self, data_matrix):
        if self.use_scaling:
            data_matrix = self.scaling(data_matrix)

        if self.use_logarithm:
            data_matrix = self.logarithm(data_matrix)

        return data_matrix
