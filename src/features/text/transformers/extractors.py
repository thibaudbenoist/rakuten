import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class YearExtractor(BaseEstimator, TransformerMixin):
    """Extracts years from a text column and returns a dataframe with the years and a boolean column indicating if the text contains a year."""
    def __init__(self, text_column):
        self.text_column = text_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        year_pattern = r"(1[6-9][0-9]{2}|20[0-1][0-9]|202[0-4])"
        years = X[self.text_column].str.extract(year_pattern, expand=False).fillna(0).astype("uint16").to_frame("year_val")
        years["has_year"] = (years["year_val"] != 0).astype("int8")
        return years
    
class NumberExtractor(BaseEstimator, TransformerMixin):
    """Extracts A Number of type 'N°' from a text column and returns a boolean column indicating if the text contains a number."""
    def __init__(self, text_column):
        self.text_column = text_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        number_pattern = r"([Nn]\s?°\s?\d+)"
        numbers = X[self.text_column].str.extract(number_pattern, expand=False).notna().rename("has_number").astype("int8")
        return numbers
    

class HashtagNumberExtractor(BaseEstimator, TransformerMixin):
    """Extracts the number of hashtags from a text column and returns a boolean column indicating if the text contains a hashtag."""
    def __init__(self, text_column):
        self.text_column = text_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        hashtag_pattern = r"(#[\d\s\-]+)"
        hashtags = X[self.text_column].str.extract(hashtag_pattern, expand=False).notna().rename("has_hashtag").astype("int8")
        return hashtags
