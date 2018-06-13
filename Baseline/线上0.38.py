from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm.sklearn import LGBMClassifier

import numpy as np
import pandas as pd


def get_ids(qids):
    ids = []
    for t_ in qids:
        ids.append(int(t_[1:]))
    return np.asarray(ids)


def get_texts(file_path, question_path):
    qes = pd.read_csv(question_path)
    file = pd.read_csv(file_path)
    q1id, q2id = file['q1'], file['q2']
    id1s, id2s = get_ids(q1id), get_ids(q2id)
    all_words = qes['words']
    texts = []
    for t_ in zip(id1s, id2s):
        texts.append(all_words[t_[0]] + ' ' + all_words[t_[1]])
    return texts


def make_submission(predict_prob):
    with open('submission.csv', 'w') as file:
        file.write(str('y_pre') + '\n')
        for line in predict_prob:
            file.write(str(line) + '\n')
    file.close()


TRAIN_PATH = 'mojing/train.csv'
TEST_PATH = 'mojing/test.csv'
QUESTION_PATH = 'mojing/question.csv'

print('Load files...')
questions = pd.read_csv(QUESTION_PATH)
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
corpus = questions['words']

print('Fit the corpus...')
vec = TfidfVectorizer()
vec.fit(corpus)

print('Get texts...')
train_texts = get_texts(TRAIN_PATH, QUESTION_PATH)
test_texts = get_texts(TEST_PATH, QUESTION_PATH)

print('Generate tfidf features...')
tfidf_train = vec.transform(train_texts[:])
tfidf_test = vec.transform(test_texts[:])


########################################
X, y = tfidf_train, train.label.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train classifier...')
clf = LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    max_depth=-1,
    num_leaves=2 ** 7 - 1,
    learning_rate=0.01,
    n_estimators=2000,

    min_split_gain=0.0,
    min_child_weight=0.001,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=888
    )

clf.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='logloss',
    early_stopping_rounds=100,
    verbose=50,
)
########################################


print('Predict...')
pred = clf.predict_proba(tfidf_test)
make_submission(pred[:, 1])

print('Complete')
