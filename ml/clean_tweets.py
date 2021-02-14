import re, string, unicodedata
import nltk
import contractions
import inflect
import numpy
import pandas
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from pandas import read_csv


def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return link_pattern.sub(r"", sample)


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = punctuation_pattern.sub(r'', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words


def preprocess(sample):
    sample = remove_URL(sample)
    sample = replace_contractions(sample)
    # Tokenize
    words = nltk.word_tokenize(sample)

    # Normalize
    return normalize(words)


link_pattern = re.compile("http\S+", flags=re.MULTILINE)
punctuation_pattern = re.compile(r'[^\w\s]')

if __name__ == "__main__":
    if True:
        print("data is already clean!")
        exit(0)

    dataset = read_csv('data/tweets.csv')
    dataset = dataset[['author', 'content']]
    dataset = dataset.fillna("")
    dataset = dataset.to_numpy()

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            # j == 0   author
            # j == 1   content
            if j == 1:  # run cleaning just for contents
                sample = remove_URL(dataset[i][j])
                sample = replace_contractions(sample)

                # Tokenize
                words = nltk.word_tokenize(sample)

                # Normalize
                words = normalize(words)
                if len(words) > 4:
                    dataset[i][j] = ' '.join(words)
                else:
                    dataset[i] = [[]]
                    numpy.delete(dataset, i)

    newDataset = pandas.DataFrame({'author': dataset[:, 0], 'content': dataset[:, 1]})
    newDataset.to_csv("data/clean_tweets-3.csv")

    print(dataset)


## test them with learned model
