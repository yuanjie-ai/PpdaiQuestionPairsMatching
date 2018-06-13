from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

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


TRAIN_PATH = 'dataset/train.csv'
TEST_PATH = 'dataset/test.csv'
QUESTION_PATH = 'dataset/question.csv'

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

print('Train classifier...')
clf = LogisticRegression()
clf.fit(tfidf_train, train['label'][:])

print('Predict...')
pred = clf.predict_proba(tfidf_test)
make_submission(pred[:, 1])

print('Complete')
