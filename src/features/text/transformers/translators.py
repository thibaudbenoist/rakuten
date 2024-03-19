import pandas as pd
import numpy as np
import time
from deep_translator import GoogleTranslator

from sklearn.base import BaseEstimator, TransformerMixin
from src.features.text.transformers.languages import LangIdDetector


class TextTranslator(BaseEstimator, TransformerMixin):
    def __init__(self, text_column=None, target_lang="fr", lang_column=None, batch_size=100, wait_time=1, verbose=0) -> None:
        self.langdetector = LangIdDetector()
        self.translators = {
            "de": GoogleTranslator(source="de", target=target_lang),
            "en": GoogleTranslator(source="en", target=target_lang),
            "fr": GoogleTranslator(source="fr", target=target_lang)
        }
        self.text_column = text_column
        self.lang_column = lang_column
        self.target_lang = target_lang
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.verbose = verbose

    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            self.text_column = X.name
        return self

    def transform(self, X):
        translations = X[self.text_column].rename("translation")
        if not self.lang_column:
            self.lang_column = "lang"
            X[self.lang_column] = self.langdetector.fit_transform(
                X[self.text_column])
        filter1 = X[self.lang_column] != self.target_lang
        filter2 = ~X[self.text_column].isna()
        lines_filter = filter1 & filter2
        indexes = X[lines_filter].index
        nb_lines = len(indexes)
        for k in range(nb_lines // self.batch_size + 1):
            batch_indexes = indexes[k *
                                    self.batch_size:min((k+1)*self.batch_size, nb_lines)]
            translations.loc[batch_indexes] = X.loc[batch_indexes].apply(lambda row: self.translate_text(
                source_lang=row.loc[self.lang_column], text=row.loc[self.text_column], target_lang=self.target_lang), axis=1)
            if self.verbose:
                print(
                    f"Traduction de {k*self.batch_size} à {min((k+1)*self.batch_size, nb_lines)} complète")
            time.sleep(self.wait_time)
        return translations

        # indexes = X[lines_filter].index
        # return indexes

        # return X.apply(lambda row: self.translate_text(row.loc[self.lang_column], row.loc[self.text_column], self.target_lang), axis=1)

    def translate_text(self, source_lang, text, target_lang):
        if pd.isna(text):
            return np.nan
        elif source_lang == target_lang:
            return text
        else:
            translated = self.translators[source_lang].translate(text)
            return translated
