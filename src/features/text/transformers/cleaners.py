import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin


class HtmlCleaner(BaseEstimator, TransformerMixin):
    """A transformer to clean HTML markups"""

    def __init__(self) -> None:
        return None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_cleaned = X.apply(lambda row: self.clean_markups(row))
        return X_cleaned
    
    @staticmethod
    def clean_markups(text):
        if isinstance(text, str):
            soup = BeautifulSoup(text, 'html.parser')
            cleaned_text = soup.get_text(separator=' ')
            return cleaned_text
        else:
            return text
        

class LxmlCleaner(BaseEstimator, TransformerMixin):
    """A transformer to clean LXML markups"""

    def __init__(self) -> None:
        return None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_cleaned = X.apply(lambda row: self.clean_markups(row))
        return X_cleaned
    
    @staticmethod
    def clean_markups(text):
        if isinstance(text, str):
            soup = BeautifulSoup(text, 'lxml')
            cleaned_text = soup.get_text(separator=' ')
            return cleaned_text
        else:
            return text
        
class TextCleaner(BaseEstimator, TransformerMixin):
    """A Transformer class to clean text"""
    def __init__(self, pattern) -> None:
        self.pattern = re.compile(pattern)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.str.replace(pat=self.pattern, repl=" ", regex=True)
    

class UrlCleaner(TextCleaner):
    """A transformer to clean URLs from text"""
    def __init__(self):
        super().__init__(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")


class FileNameCleaner(TextCleaner):
    """A transformer to clean Filenames from text"""
    def __init__(self) -> None:
        super().__init__(r'\b(?<!\d\.)\w+\.(txt|jpg|png|docx|pdf)\b')


class BadHTMLCleaner(TextCleaner):
    """A transformer to clean Bad HTML from text"""
    def __init__(self) -> None:
        super().__init__(r'nbsp|&amp|& [Nn][Bb][Ss][Pp]|')


class SpaceAroundAdder(BaseEstimator, TransformerMixin):
    """A tranformer to add space around punctuation and specific patterns"""
    def __init__(self):
        self.pattern = re.compile(r'(\d+|[-.,!¡;；:¯…„“\§«»—°•£❤☆(){}\[\]"@#$%^&*+=|<>~`‘’¬])')

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.str.replace(pat=self.pattern, repl=r" \1 ", regex=True)
    

class SpaceBeforeAdder(BaseEstimator, TransformerMixin):
    """A tranformer to add space before capital letters and specific patterns"""
    def __init__(self):
        self.pattern = re.compile(r'(?<=[a-zÀ-ÿ]|[.,!;:\§«»°])([A-Z])(?=[a-zÀ-ÿ])')

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.str.replace(pat=self.pattern, repl=r" \1", regex=True)
    

class ShortTextCleaner(BaseEstimator, TransformerMixin):
    """A transformer to remove short senseless text"""
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_cleaned = X.str.replace(pat="...", repl=".")
        X_cleaned = X_cleaned.apply(lambda row: np.nan  if isinstance(row, str) and len(row) < 3 else row)
        return X_cleaned
    
    
class ExtraSpacesCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return  X.str.replace(pat=r"\s+", repl=r" ", regex=True).str.strip()