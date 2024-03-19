import pandas as pd
import numpy as np
import spacy
import nltk
import re


def Rakuten_txt_tokenize(data, lang=None, method='spacy'):
    """
    Tokenize and lemmatize text data, returning a list of unique tokens, in the
    same order they appeared.

    Parameters:
    data (DataFrame or Series): Text data to tokenize and lemmatize.
    lang (Series, optional): Series containing language information for each
    entry.

    Returns:
    DataFrame or Series: Data with tokenized and lemmatized text.

    Usage:
    tokenized_data = Rakuten_txt_tokenize(data, language_series)
    """

    if lang is None:
        raise Exception('Invalid lang')

    # concatenating text from multiple columns if necessary
    if data.ndim > 1:
        data = data.apply(lambda row: ' '.join(
            [s for s in row if isinstance(s, str)]), axis=1)

    # joining text and language data
    data = pd.concat([data, lang], axis=1)

    if method == 'spacy':
        # Checking if spacy language data have been  downloaded
        if not spacy.util.is_package('en_core_web_sm'):
            spacy.cli.download('en_core_web_sm')

        if not spacy.util.is_package('fr_core_news_sm'):
            spacy.cli.download('fr_core_news_sm')

        if not spacy.util.is_package('de_core_news_sm'):
            spacy.cli.download('de_core_news_sm')

        # Loading spacy language models
        nlpdict = {'en': spacy.load('en_core_web_sm'),
                   'fr': spacy.load('fr_core_news_sm'),
                   'de': spacy.load('de_core_news_sm')}

        # Applying tokenisation using spacy
        data = data.apply(lambda row: tokens_from_spacy(
            row.iloc[0], row.iloc[1], nlpdict), axis=1)

    elif method == 'nltk':
        # downloading nltk ressources
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        if lang.str.len().max() == 2:
            language_mapping = {'fr': 'french',
                                'en': 'english',
                                'de': 'german'}
            data.iloc[:, 1] = data.iloc[:, 1].replace(language_mapping)

            # Applying tokenisation using spacy
            data = data.apply(lambda row: tokens_from_nltk(
                row.iloc[0], row.iloc[1]), axis=1)

    return data


def tokens_from_spacy(txt, lang, nlpdict):
    """
    Generate a list of unique word tokens from a text string using spaCy.

    Parameters:
    txt (str): Text string to tokenize.
    lang (str): Language of the text.
    nlpdict (dict): Dictionary mapping language codes to spaCy models.

    Returns:
    str: String of unique, lemmatized tokens.

    Usage:
    tokens = tokens_from_spacy(text_to_tokenize, 'en', nlp_dictionary)
    """
    if isinstance(txt, str):
        # using the appropriate language
        tokens = nlpdict[lang](txt)

        # Remove stopwords, punctuation, and perform lemmatization
        filtered_tokens = [token.lemma_.lower()
                           for token in tokens
                           if token.is_alpha
                           and not token.is_stop  # is stop is too general: parler is a stopword!!
                           and len(token) > 2
                           and any(vowel in token.text.lower() for vowel in 'aeiouyáéíóúàèìòùâêîôûäëïöü')]

        # Keeping a unique list of tokens, in the same order they appeared in
        # the text
        seen = set()
        filtered_tokens = [seen.add(token) or token
                           for token in filtered_tokens if token not in seen]

        # returning result as a single string
        return ' '.join(filtered_tokens)
    else:
        # Return nan if the input is not a string
        return np.nan


def tokens_from_nltk(txt, lang):
    """
    Generate a list of unique word tokens from a text string using NLTK.

    Parameters:
    txt (str): Text string to tokenize.
    lang (str): Language of the text.

    Returns:
    str: String of unique, lemmatized tokens.

    Usage:
    tokens = tokens_from_nltk(text_to_tokenize, 'en')
    """
    if isinstance(txt, str):
        # tokenization with the appropriate language
        tokens = nltk.tokenize.word_tokenize(txt, lang)

        # Remove stopwords, punctuation, and perform lemmatization
        stop_words = set(nltk.corpus.stopwords.words(lang))
        stemmer = nltk.stem.SnowballStemmer(lang)
        filtered_tokens = [stemmer.stem(token.lower())
                           for token in tokens
                           if token.isalpha()
                           and token.lower()
                           not in stop_words
                           and len(token) > 2
                           and any(vowel in token.lower() for vowel in 'aeiouyáéíóúàèìòùâêîôûäëïöü')]

        # Keeping a unique list of tokens, in the same order they appeared in
        # the text
        seen = set()
        filtered_tokens = [seen.add(token) or token
                           for token in filtered_tokens if token not in seen]

        # returning result as a single string
        return ' '.join(filtered_tokens)
    else:
        # Return empty lists if the input is not a string
        return np.nan


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


# add correlation heat map across categories
def Rakuten_txt_wordcount(data, nmax_words=None, Normalize=False):
    """
    Count the frequency of each word in the text data.

    Parameters:
    data (DataFrame or Series): Text data to analyze.
    nmax_words (int, optional): Maximum number of words to return.

    Returns:
    tuple: Two lists, one of word counts and the other of corresponding words,
    ordered from most to least frequent.

    Usage:
    counts, words = Rakuten_wordcount(data_with_text, 100)
    """

    # concatenating text from multiple columns if necessary
    if data.ndim > 1:
        data = data.apply(lambda row: ' '.join(
            [s for s in row if isinstance(s, str)]), axis=1)

    # Replacing NaNs by spaces
    data = data.fillna(' ')

    # Merging all strings into a single one
    data = ' '.join(data)

    # Removing non-alpha characters, converting to lower case, returning a
    # string of words
    data = re.findall(r'\b\w+\b', data.lower())

    # Counting the number of occurences of each word
    counts = dict(Counter(data).most_common(nmax_words))

    # Converting dictionary values and keys into lists
    countlist = np.array(list(counts.values()))
    wordlist = np.array(list(counts.keys()))

    # Normalizing vounts if necessary
    if Normalize:
        countlist = countlist / sum(countlist)

    # Sorting indices according to frequency and reversing to get it from most
    # to  least frequent
    idx = np.argsort(countlist)[::-1]

    # Reordering values and labels according to sorted idx
    countlist = countlist[idx].tolist()
    wordlist = wordlist[idx].tolist()

    return countlist, wordlist