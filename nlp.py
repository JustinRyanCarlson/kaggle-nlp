import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sample = pd.read_csv('./data/sample_submission.csv')

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author.values)

xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)

tfidf = TfidfVectorizer(min_df=3,  max_features=None, 
                        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                        ngram_range=(1, 3) , use_idf=1 ,smooth_idf=1 , sublinear_tf=1,
                        stop_words='english')

tfidf.fit(list(xtrain) + list(xvalid))
xtrain_tfidf = tfidf.transform(xtrain)
xvalid_tfidf = tfidf.transform(xvalid)

clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfidf, ytrain)
predict = clf.predict_proba(xvalid_tfidf)
print('TFIDF/LogisticRegression logless: {}'.format(multiclass_logloss(yvalid, predict)))

count_vec = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                        ngram_range=(1, 3), stop_words = 'english')
count_vec.fit(list(xtrain) + list(xvalid))
xtrain_count = count_vec.transform(xtrain)
xvalid_count = count_vec.transform(xvalid)

clf = LogisticRegression(C=1.0)
clf.fit(xtrain_count, ytrain)
predict = clf.predict_proba(xvalid_count)
print('count/LogisticRegression logless: {}'.format(multiclass_logloss(yvalid, predict)))

clf = MultinomialNB()
clf.fit(xtrain_tfidf, ytrain)
predict = clf.predict_proba(xvalid_tfidf)
print('TFIDF/MultinomialNB logless: {}'.format(multiclass_logloss(yvalid, predict)))

clf = MultinomialNB()
clf.fit(xtrain_count, ytrain)
predict = clf.predict_proba(xvalid_count)
print('count/MultinomialNB logless: {}'.format(multiclass_logloss(yvalid, predict)))

# SLOW and performs badly
# 
# svd = decomposition.TruncatedSVD(n_components=200)
# svd.fit(xtrain_tfidf)
# xtrain_svd = svd.transform(xtrain_tfidf)
# xvalid_svd = svd.transform(xvalid_tfidf)

# scl = preprocessing.StandardScaler()
# scl.fit(xtrain_svd)
# xtrain_svd_scl = scl.transform(xtrain_svd)
# xvalid_svd_scl = scl.transform(xvalid_svd)

# clf = SVC(C=1.0, probability=True)
# clf.fit(xtrain_svd_scl, ytrain)
# predict = clf.predict_proba(xvalid_svd_scl)
# print('TFIDF/SVD/SVC logless: {}'.format(multiclass_logloss(yvalid, predict)))

