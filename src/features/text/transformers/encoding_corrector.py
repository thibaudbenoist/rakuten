import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.features.text.helpers import ENCODING_REPLACEMENT_DICT, CLASSIC_PATTERNS_DICT
from src.features.text.transformers.languages import LangIdDetector
from spellchecker import SpellChecker
from functools import lru_cache

def Rakuten_txt_fixencoding(data, lang):
    """
    Cleans up text data by correcting badly encoded words.

    This function joins text data with language information, applies encoding corrections,
    and utilizes spell checking for various languages to correct misspelled words.
    It specifically targets issues with special characters like '?', '¿', 'º', '¢', '©', and '́'.

    Parameters:
    data (DataFrame): A pandas DataFrame containing text data to be cleaned.
    lang (Series): A pandas Series containing language information for each row in the data.

    Returns:
    DataFrame: The cleaned DataFrame with bad encodings corrected and language column dropped.

    Usage Example:
    df = Rakuten_txt_cleanup(data[['designation', 'description']],
                                       data['language'])
    """
    # joining text and language data
    data = pd.concat([data, lang.rename('lang')], axis=1)

    # Correction of bad encoding relies in part on spell checker, to correct
    # misspelled words.
    spellers = {'fr': SpellChecker(language='fr'),
                'en': SpellChecker(language='en'),
                'de': SpellChecker(language='de')}

    for col in data.columns[:-1]:
        data.loc[:, col] = data.apply(
            lambda row: txt_fixencoding(row[col], row['lang'], spellers), axis=1)

    return data.drop(columns='lang')


def txt_fixencoding(txt, lang, spellers):
    """
    Corrects badly encoded words within a given text string.

    This function applies multiple regex substitutions to fix common encoding issues 
    in text data. It handles duplicates of badly encoded characters, replaces specific 
    incorrectly encoded words with their correct forms, and utilizes a spell checker for further corrections. 
    The function also handles special cases of encoding errors after certain character sequences.

    Parameters:
    txt (str): The text string to be cleaned.
    lang (str): The language of the text, used for spell checking.
    spellers (dict): A dictionary of SpellChecker instances for different languages.

    Returns:
    str: The cleaned text string with encoding issues corrected.

    Usage Example:
    ----------------
    # Spell checkers for different languages
    spellers = {'fr': SpellChecker(language='fr'),
                'en': SpellChecker(language='en'),
                'de': SpellChecker(language='de')}

    # Correct the encoding in the text
    corrected_text = txt_fixencoding(example_text, language, spellers)
    """

    # Not sure why but the following doesn't work at once so we do it again
    # (It is quite common in the data set...)
    pattern = re.escape('durabilitéavec')
    txt = re.sub(pattern, 'durabilité avec', txt)
    pattern = re.escape('cahiercaractère')
    txt = re.sub(pattern, 'cahier caractère', txt)
    pattern = re.escape('Cahieremblème')
    txt = re.sub(pattern, 'cahier emblème', txt)


    # Finding all remaining words with special characters at the start, end or
    # within
    pattern = r'\b\w*[\?¿º¢©́]\w*(?:[\?¿º¢©́]+\w*)*|\b\w*[\?¿º¢©́]|[\?¿º¢©́]\w*(?:[\?¿º¢©́]+\w*)*'
    badword_list = re.findall(pattern, txt)
    # Since this ends up with some special characters alone (for instance à
    # would become single ?), we make sure they are the last to be corrected.
    # Otherwise, the other motifs wouldn't be detectable anymore in the next
    # loop
    badword_list.sort(key=len, reverse=True)

    # correction function with lru_cache to enable caching
    @lru_cache(maxsize=1000)
    def cached_spell_correction(word, language):
        return spellers[lang].correction(word)

    # Replacing each of these word by the correction from the spell checker (if
    # available), if it is different from the original word without special
    # character (in case this character is an actual punctuation)
    for badword in badword_list:
        badword = badword.lower()
        badword_corrected = cached_spell_correction(badword, lang)
        badword_cleaned = re.sub(r'[^a-zA-Z0-9]', '', badword)
        if badword_corrected and badword_corrected != badword_cleaned:
            pattern = re.escape(badword)
            #txt = txt.replace(badword, badword_corrected)
            txt = re.sub(pattern, badword_corrected, txt, flags=re.IGNORECASE)

    # for debugging purpose
    pattern = r'\b\w*[\?¿º¢©́]\w*(?:[\?¿º¢©́]+\w*)*|\b\w*[\?¿º¢©́]|[\?¿º¢©́]\w*(?:[\?¿º¢©́]+\w*)*'
    badword_list = re.findall(pattern, txt)

    # at the end we remove all remaining special characters except ? (as those)
    # at the end of words may correspond to actual punctuations
    pattern = r'[\¿º¢©́]'
    txt = re.sub(pattern, ' ', txt)

    # Adding a space after ? if necessary
    pattern = r'\?(?!\s)'
    txt = re.sub(pattern, '? ', txt)

    return txt

class EncodingIsolator(BaseEstimator, TransformerMixin):
    """A text transformer to replace duplicates of badly encoded markers and some weird combinations with a single one"""
    def __init__(self) -> None:
        self.pattern = re.compile(r'([?¿º¢©́Ã])\1+|[ã¢©́]\1+')
        self.special_sequence_pattern = re.compile(r'[å¿]')

    def fit(self, X, y=None):
        return self    

    def transform(self, X):
        # Replace duplicates of badly encoded markers
        X_cleaned = X.str.replace(pat=self.pattern, repl=r'\1', regex=True)
        X_cleaned = X_cleaned.str.replace(pat=self.special_sequence_pattern, repl='¿', regex=True)
        return X_cleaned
    
class SpecificIsolator(BaseEstimator, TransformerMixin):
    """"""
    def __init__(self) -> None:
        self.pattern = re.compile(r"[ã¢©́]+")

    def fit(self, X, y=None):
        return self    

    def transform(self, X):
        # Replace duplicates of badly encoded markers
        return X.str.replace(pat=self.pattern,repl='¿', regex=True)
    

class ExcessPunctuationRemover(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.pattern = re.compile(r"[?\s:;\/!¿]{3,}")

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_cleaned = X.str.replace(pat=self.pattern, repl=" . ", regex=True)
        return X_cleaned
    

class HandMadeCorrector(BaseEstimator, TransformerMixin):
    """"""
    def __init__(self) -> None:
        self.replacement_dict = ENCODING_REPLACEMENT_DICT

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_corrected = X
        for badword, correction in self.replacement_dict.items():
            X_corrected = X_corrected.str.replace(pat=badword, repl=correction, case=False)
        return X_corrected
    

class EncodingApostropheCorrector(BaseEstimator, TransformerMixin):
    """"""
    def __init__(self) -> None:
        self.pattern=re.compile(r'\b([sdlcn])[¿?](?=[aeiouyhAEIOUYHé])')

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_corrected = X.str.replace(pat=self.pattern, repl=r"\1'", regex=True)
        return X_corrected


class PronounsApostropheCorrector(BaseEstimator, TransformerMixin):
    """"""
    def __init__(self) -> None:
        self.pattern=re.compile( r'\bqu[¿?]')

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_corrected = X.str.replace(pat=self.pattern, repl="qu'", regex=True)
        return X_corrected
    

class LowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.str.lower()
    

class ClassicPatternsCorrector(BaseEstimator, TransformerMixin):
    """"""
    def __init__(self) -> None:
        self.replacement_dict = CLASSIC_PATTERNS_DICT

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_corrected = X
        for pattern, correction in self.replacement_dict.items():
            X_corrected = X_corrected.str.replace(pat=pattern, repl=correction, case=False)
        return X_corrected
    
    
class DoubleEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.pattern = re.compile( r'â¿')

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.str.replace(pat=self.pattern, repl="¿", regex=True)
    

class NumberEncodingCorrector(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.pattern = re.compile(r"¿(\d+)|(\d+)¿")

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return  X.str.replace(pat=self.pattern, repl=r"\1", regex=True)
    
class BadWordsDetector(BaseEstimator, TransformerMixin):
    """"""
    def __init__(self) -> None:
        self.pattern = re.compile( r'\w*[å¿]\w*')
        self.lang_detector = LangIdDetector()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        bad_words = X.str.findall(pat=self.pattern)
        languages = self.lang_detector.fit_transform(X)
        return pd.DataFrame({
            "text":X, 
            "badwords": bad_words, 
            "lang": languages 
            })
    
class BadWordsCorrector(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.spellers = {
            "fr": SpellChecker(language="fr"),
            "en": SpellChecker(language="en"),
            "de": SpellChecker(language="de"),
        }

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        corrected_text = X.apply(lambda row: self.correct_text(row["text"], row["badwords"], row["lang"]),  axis=1)
        #rows = [x for x in X.itertuples()]
        #corrected_text = [self.correct_text(row.text, row.badwords, row.lang) for row in rows]
        #corrected_text = pd.Series(corrected_text)
        return corrected_text

    @lru_cache(maxsize=1000)
    def correct_word(self, word, lang):
        correction = self.spellers[lang].correction(word)
        return correction
        
    def correct_text(self, text, badwords, lang):
        if isinstance(text, str):
            for word in badwords:
                try:
                    correction = self.correct_word(word=word, lang=lang)
                    text = re.sub(word, correction, text)
                except(TypeError):
                    print(f"Badword: {word}\n Text to correct: {text}")
                    
                    #raise TypeError
        return text


class PySpellCorrector(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.bwd = BadWordsDetector()
        self.bwc = BadWordsCorrector()


    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_info = self.bwd.fit_transform(X)
        X_corrected = self.bwc.fit_transform(X_info)
        return X_corrected
    
class AccentCorrector(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.pattern = re.compile(r"ã¿")

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.str.replace(pat=self.pattern, repl="é", regex=True)

    


class SpaceReplacement(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.pattern = re.compile(r"¿")

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.str.replace(pat=self.pattern, repl=" ", regex=True)