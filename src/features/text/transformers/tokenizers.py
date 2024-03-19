import numpy as np
import pandas as pd
import nltk
import spacy

from sklearn.base import BaseEstimator, TransformerMixin

from src.features.text.transformers.languages import LangIdDetector


def merge_tokens(*args):
    """ merge two strings, keeping a unique list of the words, in the order
    they appeared"""
    # Replacing non-string inputs with empty string
    args = [arg if isinstance(arg, str) else '' for arg in args]

    # joining input strings
    alltokens = (' '.join(args)).split()

    # Keeping a unique list of tokens, in the same order they appeared in
    # the text
    seen = set()
    alltokens = [seen.add(token) or token
                 for token in alltokens if token not in seen]

    return ' '.join(alltokens)


class SpacyTokenizer(BaseEstimator, TransformerMixin):

    def __init__(self, text_column=None, lang_column=None, keep_stopwords=False, unique_tokens=True) -> None:
        self.langdetector = LangIdDetector()
        self.unique_tokens = unique_tokens
        self.keep_stopwords = keep_stopwords
        self.text_column = text_column
        self.lang_column = lang_column

        if not spacy.util.is_package('en_core_web_sm'):
            spacy.cli.download('en_core_web_sm')

        if not spacy.util.is_package('fr_core_news_sm'):
            spacy.cli.download('fr_core_news_sm')

        if not spacy.util.is_package('de_core_news_sm'):
            spacy.cli.download('de_core_news_sm')

        # Loading spacy language models
        self.tokenizers = {
            'en': spacy.load('en_core_web_sm'),
            'fr': spacy.load('fr_core_news_sm'),
            'de': spacy.load('de_core_news_sm')
            }
        
    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            self.text_column = X.name
        return self
    
    def transform(self, X):
        if not self.lang_column:
            self.lang_column = "lang"
            X[self.lang_column] = self.langdetector.fit_transform(X[self.text_column])
        return X.apply(lambda row: self.tokenize(row[self.text_column], row[self.lang_column]), axis=1)
    
    def tokenize(self, text, lang):
        if isinstance(text, str):
            tokens = self.tokenizers[lang](text)
            if not self.keep_stopwords:
                filtered_tokens = [token.lemma_.lower()
                            for token in tokens
                            if token.is_alpha
                            and len(token) > 2
                            and not token.is_stop
                            and any(vowel in token.text.lower() for vowel in 'aeiouyáéíóúàèìòùâêîôûäëïöü')]
            else:
                filtered_tokens = [token.lemma_.lower()
                            for token in tokens
                            if token.is_alpha
                            and len(token) > 2
                            and any(vowel in token.text.lower() for vowel in 'aeiouyáéíóúàèìòùâêîôûäëïöü')]
            if self.unique_tokens:
                filtered_tokens = list(set(filtered_tokens))
            return " ".join(filtered_tokens)
        else:
            return np.nan


class NLTKTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, text_column=None, lang_column=None, keep_stopwords=False, unique_tokens=True) -> None:
        self.langdetector = LangIdDetector()
        self.unique_tokens = unique_tokens
        self.keep_stopwords = keep_stopwords
        self.text_column = text_column
        self.lang_column = lang_column

        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        self.lang_mapper = {
            'fr': 'french',
            'en': 'english',
            'de': 'german'
            }
        


    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            self.text_column = X.name
        return self
    
    def transform(self, X):
        if not self.lang_column:
            self.lang_column = "lang"
            X[self.lang_column] = self.langdetector.fit_transform(X[self.text_column])
        X["token_lang"] = X[self.lang_column].map(self.lang_mapper)
        return X.apply(lambda row: self.tokenize(row[self.text_column], row["token_lang"]), axis=1)
    
    def tokenize(self, text, lang):
        if isinstance(text, str):
        # tokenization with the appropriate language
            tokens = nltk.tokenize.word_tokenize(text, lang)
            # Remove stopwords, punctuation, and perform lemmatization
            stop_words = set(nltk.corpus.stopwords.words(lang))
            stemmer = nltk.stem.SnowballStemmer(lang)
            if not stop_words:
                filtered_tokens = [stemmer.stem(token.lower())
                                for token in tokens
                                if token.isalpha()
                                and token.lower()
                                not in stop_words
                                and len(token) > 2
                                and any(vowel in token.lower() for vowel in 'aeiouyáéíóúàèìòùâêîôûäëïöü')]
            else:
                filtered_tokens = [stemmer.stem(token.lower())
                                for token in tokens
                                if token.isalpha()
                                and token.lower()
                                and len(token) > 2
                                and any(vowel in token.lower() for vowel in 'aeiouyáéíóúàèìòùâêîôûäëïöü')]
            if self.unique_tokens:
                filtered_tokens = list(set(filtered_tokens))
            return " ".join(filtered_tokens)
        else:
            return np.nan


    