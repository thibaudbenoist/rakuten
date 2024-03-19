import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import py3langid as langid
from spellchecker import SpellChecker
from transformers import BertTokenizer



class LangIdDetector(BaseEstimator, TransformerMixin):
    """
    A transformer to extract language from a text
    """
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        langid.set_languages(['fr', 'en', 'de'])
        X_raw = X.fillna("Français")
        lang = X_raw.apply(lambda row: langid.classify(row)[0])
        return lang


class PyspellDetector(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_raw = X.fillna("Français")
        spell_fr = SpellChecker(language='fr', distance=1)
        spell_en = SpellChecker(language='en', distance=1)
        spell_de = SpellChecker(language='de', distance=1)

        err_fr = X_raw.apply(lambda row: len(spell_fr.known(row.split())))
        err_en = X_raw.apply(lambda row: len(spell_en.known(row.split())))
        err_de = X_raw.apply(lambda row: len(spell_de.known(row.split())))
        lang = pd.concat([err_fr.rename('fr'), err_en.rename(
            'en'), err_de.rename('de')], axis=1)
        lang = lang.idxmax(axis=1)
        return lang

class BertDetector(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_raw = X.fillna("Français")
        tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer_fr = BertTokenizer.from_pretrained(
            'dbmdz/bert-base-french-europeana-cased')
        tokenizer_de = BertTokenizer.from_pretrained('bert-base-german-cased')

        err_fr = X_raw.apply(lambda row: ' '.join(
            tokenizer_fr.convert_ids_to_tokens(tokenizer_fr(row)['input_ids'])))
        err_en = X_raw.apply(lambda row: ' '.join(
            tokenizer_en.convert_ids_to_tokens(tokenizer_en(row)['input_ids'])))
        err_de = X_raw.apply(lambda row: ' '.join(
            tokenizer_de.convert_ids_to_tokens(tokenizer_de(row)['input_ids'])))
        lang = pd.concat([err_fr.rename('fr'), err_en.rename(
            'en'), err_de.rename('de')], axis=1)
        lang = lang.idxmin(axis=1)
        return lang
