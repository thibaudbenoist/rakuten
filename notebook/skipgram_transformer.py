import numpy as np
from sklearn.base import TransformerMixin
from gensim.models import Word2Vec


class SkipGramTransformer(TransformerMixin):
    """
    SkipGramTransformer
    
    This class is a transformer that uses the SkipGram Word2Vec model to transform a list of sentences into a list of vectors.

    Parameters
    ----------
    vector_size : int, default=100
        The size of the word vectors.

    window : int, default=5
        The maximum distance between the current and predicted word within a sentence.

    min_count : int, default=5
        Ignores all words with total frequency lower than this.

    workers : int, default=4
        The number of workers to use for training the model.


    Attributes
    ----------
    w2v_model : Word2Vec
        The Word2Vec model used to transform the sentences into vectors.    


    Methods
    -------
    fit(X, y=None)
        Fit the Word2Vec model to the sentences in X.

    transform(X)
        Transform the sentences in X into vectors using the Word2Vec model.

    vectorize(sentence)
        Transform a sentence into a vector using the Word2Vec model.


    Example
    -------
    >>> from skipgram_transformer import SkipGramTransformer
    >>> sentences = ["this is a sentence", "this is another sentence"]
    >>> transformer = SkipGramTransformer()
    >>> transformer.fit(sentences)
    >>> vectors = transformer.transform(sentences)
    >>> print(vectors)
    """
    def __init__(self, vector_size=100, window=5, min_count=5, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.w2v_model = None

    def fit(self, X, y=None):
        sentences = [sentence.split() for sentence in X]
        self.w2v_model = Word2Vec(sentences, window=self.window, min_count=self.min_count, workers=self.workers, sg=1, vector_size=self.vector_size)
        return self

    def transform(self, X):
        return np.array([self.vectorize(sentence) for sentence in X])

    def vectorize(self, sentence):
        words = sentence.split()
        words_vecs = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
        if len(words_vecs) == 0:
            return np.zeros(self.w2v_model.wv.vector_size)
        words_vecs = np.array(words_vecs)
        return words_vecs.mean(axis=0)
