import sys

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import ascii_uppercase, ascii_lowercase, punctuation
from pandas import DataFrame, crosstab, read_csv
from collections import Counter
from math import log, inf
from numpy import vectorize, diag
from functools import partial
from datetime import datetime
from sklearn.metrics import matthews_corrcoef
import logging


class Classifier(object):
    class Vectorize(vectorize):
        def __get__(self, obj, objtype):
            return partial(self.__call__, obj)

    def __init__(self, featureLabels):
        self.__stopWords__ = set(stopwords.words('english'))
        self.__transTable__ = str.maketrans(ascii_uppercase + punctuation + "1234567890",
                                            ascii_lowercase + ' ' * (len(punctuation) + 10), '')
        self.__lemmatizer__ = WordNetLemmatizer()
        self.__featureLabels__ = featureLabels
        self.__probabilities__ = {}
        self.__probabilitiesOfLabels__ = {}

    def __clean__(self, text):
        text = text.replace("https://www.huffingtonpost.com/entry/", " ")
        text = text.translate(self.__transTable__)
        tokens = word_tokenize(text)
        removedStopWords = [word for word in tokens if word not in self.__stopWords__]
        normalized = [self.__lemmatizer__.lemmatize(word) for word in removedStopWords]
        return normalized

    @Vectorize
    def __predictLabel__(self, line):
        line = self.__clean__(line)
        maxProb = -inf, ''
        for label in self.__featureLabels__:
            min_prob = min(self.__probabilities__[label].values())
            maxProb = max(
                maxProb,
                (
                    self.__probabilitiesOfLabels__[label] +
                    sum([self.__probabilities__[label].get(word, min_prob) for word in line]),
                    label
                )
            )
        return maxProb[1]

    def fit(self, train):
        train = train[train['short_description'].notnull()]
        numOfTotalRows = train[train['category'].isin(self.__featureLabels__)].shape[0]
        for label in self.__featureLabels__:
            data = train[train['category'] == label][['short_description', 'headline', 'authors', 'link']]
            merged = " ".join(
                data['short_description'].tolist() +
                data['headline'].tolist() +
                data['authors'].tolist() +
                data['link'].tolist()
            )
            parsed = self.__clean__(merged)
            total_label = len(parsed)

            self.__probabilities__[label] = {i: log(float(j / total_label)) for i, j in dict(Counter(parsed)).items()}
            self.__probabilitiesOfLabels__[label] = log(float(data.shape[0] / numOfTotalRows))

    def predict(self, data):
        data = data[data['short_description'].notnull()]
        data['temp'] = data['short_description'] + " " + data['headline'] + " " + data['authors'] + " " + data['link']

        return self.__predictLabel__(data['temp'].tolist())


class TrainPredictSelector(object):

    def __init__(self, data, category, oversample):
        self.dataset = data[data['short_description'].notnull()]
        self.category = category
        self.oversample = oversample

    def __overSample__(self, train, test=True):
        max_count = max(train['category'].value_counts())
        for label in self.category:
            count = train[train['category'] == label].shape[0]
            if count < max_count:
                train = train.append(train[train['category'] == label].sample(n=max_count - count, replace=True))
        max_count = max(test['category'].value_counts())
        for label in self.category:
            count = test[test['category'] == label].shape[0]
            if count < max_count:
                test = test.append(test[test['category'] == label].sample(n=max_count - count, replace=True))
        return train, test

    def prepareTrainAndPredict(self):
        data = self.dataset
        # ['category', 'short_description', 'authors', 'headline', 'link']
        trainSet = DataFrame()
        predictSet = DataFrame()

        # ['author', 'content']
        tweets = read_csv('data/tweets.csv')
        tweets.columns = ['', 'authors', 'short_description']
        tweets['headline'] = ""
        tweets['link'] = ""
        tweets['category'] = ""

        for label in self.category:
            type1 = data[data['category'] == label]
            trainSet = trainSet.append(type1.iloc[:, :])
            predictSet = predictSet.append(tweets.iloc[:, :])

        if self.oversample:
            return self.__overSample__(trainSet[trainSet['category'].notnull()], predictSet)
        else:
            return trainSet[trainSet['category'].notnull()], predictSet


class Evaluator(object):
    def __init__(self, predicts, answers):
        pairs = DataFrame(data={'predict': predicts, 'actual': answers})
        self.cnfMatrix = crosstab(pairs['predict'], pairs['actual'])

    def accuracy(self):
        return diag(self.cnfMatrix).sum() / self.cnfMatrix.values.sum()

    def cnfusionMatrix(self):
        return self.cnfMatrix

    def precision(self):
        return DataFrame(diag(self.cnfMatrix) / self.cnfMatrix.sum(axis=1), index=self.cnfMatrix.columns,
                         columns=["precision"])
    def recall(self):
        return DataFrame(diag(self.cnfMatrix) / self.cnfMatrix.sum(axis=0), index=self.cnfMatrix.columns,
                         columns=["recall"])

    def resultTable(self, title):
        recall = diag(self.cnfMatrix) / self.cnfMatrix.sum(axis=0)
        precision = diag(self.cnfMatrix) / self.cnfMatrix.sum(axis=1)
        f1 = 2 * (precision * recall) / (precision + recall)
        df = DataFrame(columns=self.cnfMatrix.columns)
        df.columns.name = title
        df = df.append(recall, ignore_index=True)
        df = df.append(precision, ignore_index=True)
        df = df.append(f1, ignore_index=True)
        df = df.append({self.cnfMatrix.columns[0]: self.accuracy()}, ignore_index=True)
        df.index = ['recall', 'precision', 'f1', 'accuracy']
        return df.fillna('')


categories = [
    "POLITICS",
    "TRAVEL",
    "BUSINESS",
    "COMEDY",
    "SPORTS",
    "HOME & LIVING",
    "WEDDINGS",
    "DIVORCE",
    "CRIME",
    "MEDIA",
    "TECH",
    "HEALTHY",
    "STYLE & BEAUTY",
    "FOOD & DRINK",
    "ARTS & CULTURE",
    "EDUCATION & SCIENCE",
]

startedAt = datetime.now()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

dataset = read_csv('data/news.csv')
dataset = dataset[['category', 'short_description', 'authors', 'headline', 'link']]
dataset = dataset.fillna("")
dataset = dataset.sample(frac=1)  # shuffle dataset to avoid biased

logging.info("===============")
logging.info("preparing dataset done: ")
logging.info(datetime.now() - startedAt)

ts = TrainPredictSelector(dataset, categories, oversample=False)
trainData, testData = ts.prepareTrainAndPredict()

###########################
# testData = testData[:10]
###########################

cf = Classifier(categories)
cf.fit(trainData)

logging.info("===============")
logging.info("training done: ")
logging.info(datetime.now() - startedAt)

predicts = cf.predict(testData)

logging.info("===============")
logging.info("predicts done: ")
logging.info(datetime.now() - startedAt)

testData["predict"] = predicts
testData.drop(["headline", "link", "category"], inplace=True, axis=1)
testData.to_csv('data/predicted-tweets.csv')

logging.info("DONE")
logging.info(datetime.now() - startedAt)
logging.info(ev.cnfusionMatrix())
logging.info(ev.resultTable("RESULTS"))
logging.info("MCC: ")  # y_true, y_pre
logging.info(matthews_corrcoef(test['category'].tolist(), predicts))
logging.info("done: ", datetime.now() - startedAt)
logging.info(datetime.now() - startedAt)
