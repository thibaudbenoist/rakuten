import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

class WordCounter(BaseEstimator, TransformerMixin):
    def __init__(self, text_column, max_words=None, normalize=False) -> None:
        self.max_words = max_words
        self.normalize = normalize
        self.text_column = text_column
        self.pattern = re.compile(r'\b\w+\b')

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        texts = " ".join(X[self.text_column].fillna(" "))
        split_text = re.findall(self.pattern, texts.lower())
        counts = dict(Counter(split_text).most_common(self.max_words))
        countlist = np.array(list(counts.values()))
        wordlist = np.array(list(counts.keys()))
        if self.normalize:
            countlist = countlist / sum(countlist)
        idx = np.argsort(countlist)[::-1]
        countlist = countlist[idx].tolist()
        wordlist = wordlist[idx].tolist()

        return countlist, wordlist