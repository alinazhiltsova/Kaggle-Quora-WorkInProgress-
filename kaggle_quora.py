# -*- coding: utf-8 -*-
"""Kaggle Quora

Original file is located at
    https://colab.research.google.com/drive/1x3ZYzRz7sS4VX7i98ZX4MTtJQZTFjIHf
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

from google.colab import files
uploaded = files.upload()

from google.colab import files
uploaded = files.upload()

import io
df = pd.read_csv(io.BytesIO(uploaded['train.csv']))

df_test = pd.read_csv(io.BytesIO(uploaded['test.csv']))

df_test.shape



print('Total N of pairs for training: {}'.format(len(df)))

print('Duplicate pairs: {}%'.format(round(df['is_duplicate'].
                                          mean()*100,2)))

qids = pd.Series(df['qid1'].tolist()+df['qid2'].tolist())

print('Total N of questions in training data: {}'
.format(len(np.unique(qids))))

print('N of Q that appear several times: {}'.format
      (np.sum(qids.value_counts() > 1)))

tr_questns = pd.Series(df['question1'].tolist() + df['question2'].tolist()).astype(str)

test_questns = pd.Series(df_test['question1'].tolist()+df_test['question2'].tolist()).astype(str)

dist_train = tr_questns.apply(len)
dist_test = test_questns.apply(len)

#for visualisation

pal = sns.color_palette()

plt.figure(figsize=(15,10))
plt.hist(dist_train, bins = 200, range=[0,200], 
         color=pal[2],normed = True, label='train')
plt.hist(dist_test, bins= 200, range=[0,200], color=pal[1], 
         normed=True, alpha=0.5, label='test')
plt.title('Normalised histogram of character count in Qs', fontsize=15)
plt.legend()
plt.xlabel('N of characters', fontsize=15)
plt.ylabel('Probability', fontsize=15)

def squares(n=10):
  print('gen squares'.format(n ** 2))
  for i in range(1, n+1):
    yield i ** 2

def squaress(n=10):
  print('gen squares'.format(n ** 2))
  for i in range(1, n+1):
    return i ** 2

squaress()



for x in gen:
  print(x, end=' ')


stops = stopwords.words('english')

def word_match_share(row):
  q1words = {}
  q2words = {}
  for word in str(row[1]).lower().split():
    if word not in stops:
      q1words[word] = 1
  for word in str(row[2]).lower().split():
    if word not in stops:
      q2words[word] = 1
  if len(q1words) == 0 or len(q2words) == 0:
    return 0
  
  shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
  shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

  R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
  return R



plt.figure(figsize=(15, 5))
train_word_match = df.apply(word_match_share, axis=1, raw=True)
plt.hist(train_word_match[df['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
plt.hist(train_word_match[df['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)

from collections import Counter

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(tr_questns)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

print('Most common words and weights: \n')
print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
print('\nLeast common words and weights: ')
(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])

def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row[1]).lower().split():
      if word not in stops:
        q1words[word] = 1
    for word in str(row[2]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

plt.figure(figsize=(15, 5))
tfidf_train_word_match = df.apply(tfidf_word_match_share, axis=1, raw=True)
plt.hist(tfidf_train_word_match[df['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')
plt.hist(tfidf_train_word_match[df['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over tfidf_word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)

from sklearn.metrics import roc_auc_score
print('Original AUC:', roc_auc_score(df['is_duplicate'], train_word_match))
print('TFIDF AUC:', roc_auc_score(df['is_duplicate'], tfidf_train_word_match.fillna(0)))

x_train = pd.DataFrame()
x_test = pd.DataFrame()

x_train['word_match'] = train_word_match
x_train['tfidf_word_match'] = tfidf_train_word_match

x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)

x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)

y_train = df['is_duplicate'].values

x_test.shape



pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]

p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

len(x_train)

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

import xgboost as xgb

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

from google.colab import files
uploaded = files.upload()

subs = pd.read_csv(io.BytesIO(uploaded['sample_submission.csv']))

subs.head()
p_test_1 = p_test[:2345796]
subs.loc[:, 'is_duplicate'] = p_test_1
subs.to_csv('subs.csv', index=False)
