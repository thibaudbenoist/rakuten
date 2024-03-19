from sklearn.base import BaseEstimator, TransformerMixin

class TextMerger(BaseEstimator, TransformerMixin):
    """Merges the designation and description columns into a single column."""
    def __init__(self, designation_column: str, description_column: str, merged_column: str):
        self.designation_column = designation_column
        self.description_column = description_column
        self.merged_column = merged_column
    
    
    def fit(self, X, y= None):
        return self

    def transform(self, X):
        merged_column =X[self.designation_column].fillna("") + ' ' + X[self.description_column].fillna("")
        merged_column = merged_column.str.strip().rename(self.merged_column)
        return merged_column