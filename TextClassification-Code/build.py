import nltk
import unicodedata

from loader import CorpusLoader
from reader import PickledCorpusReader
from nltk.stem.porter import *
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def identity(words):
    return words


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english'):
        self.language = language
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.stemmer = PorterStemmer()

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        return [
            self.stem(token).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def stem(self, token):
        return self.stemmer.stem(token)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.normalize(document[0])


def create_pipeline(estimator, reduction=False):
    steps = [
        ('normalize', TextNormalizer()),
        ('vectorize', TfidfVectorizer(
            tokenizer=identity, preprocessor=None, lowercase=False
        ))
    ]

    if reduction:
        steps.append((
            'reduction', TruncatedSVD(n_components=10000)
        ))

    # Add the estimator
    steps.append(('classifier', estimator))
    return Pipeline(steps)


labels = ["books", "cinema", "design", "sports", "tech"]
reader = PickledCorpusReader('../corpus')
loader = CorpusLoader(reader, 5, shuffle=True, categories=labels)

models = []
for form in (SVC, DecisionTreeClassifier, RandomForestClassifier):
    models.append(create_pipeline(form(), False))
