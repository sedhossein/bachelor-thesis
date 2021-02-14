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
            # print({i + " " + j for i, j in dict(name = "John", age = "36", country = "Norway").items()})
            ## {'name John', 'country Norway', 'age 36'}
            # print({i: i + " " + j for i, j in dict(name = "John", age = "36", country = "Norway").items()})
            ## {'name': 'name John', 'age': 'age 36', 'country': 'country Norway'}
            self.__probabilities__[label] = {i: log(float(j / total_label)) for i, j in dict(Counter(parsed)).items()}
            self.__probabilitiesOfLabels__[label] = log(float(data.shape[0] / numOfTotalRows))

    def predict(self, data, final_test=False):
        ## First implementation
        data = data[data['short_description'].notnull()]
        data['temp'] = data['short_description'] + " " + data['headline'] + " " + data['authors'] + " " + data['link']
        if not final_test:
            data = data[data['category'].isin(self.__featureLabels__)]

        return self.__predictLabel__(data['temp'].tolist())

class TrainTestSelector(object):
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

    def divideTestAndTrain(self, train=0.9):
        data = self.dataset
        trainset = DataFrame().reindex_like(data)
        testset = DataFrame().reindex_like(data)
        for label in self.category:
            type1 = data[data['category'] == label]
            trainset = trainset.append(type1.iloc[:int(train * type1.shape[0]), :])
            testset = testset.append(type1.iloc[int(train * type1.shape[0]):, :])
        if self.oversample:
            return self.__overSample__(trainset[trainset['category'].notnull()], testset[testset['category'].notnull()])
        else:
            return trainset[trainset['category'].notnull()], testset[testset['category'].notnull()]

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
        f1 = 2*(precision*recall)/(precision+recall)
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

dataset = read_csv('data/news-clean-category-1.csv')
dataset = dataset[['category', 'short_description', 'authors', 'headline', 'link']]
dataset = dataset.fillna("")
dataset = dataset.sample(frac=1)

print("===============================")
print("preparing dataset done: ", datetime.now() - startedAt)

ts = TrainTestSelector(dataset, categories, oversample=True)
train, test = ts.divideTestAndTrain()
cf = Classifier(categories)
cf.fit(train)

print("===============================")
print("training done: ", datetime.now() - startedAt)

predicts = cf.predict(test)

print("===============================")
print("predicts done: ", datetime.now() - startedAt)


ev = Evaluator(predicts, test['category'].tolist())

print(ev.cnfusionMatrix())
print("===============================")
print(ev.resultTable("RESULTS"))
print("===============================")
print("MCC") # y_true, y_pre
print(matthews_corrcoef(test['category'].tolist(), predicts)) # y_true, y_pre
print("===============================")
print("done: ", datetime.now() - startedAt)