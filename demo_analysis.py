"""
Demo 
Batch Learning for Publication Classification

Usage:

Options:

Examples:

License:

Copyright (c) 2015 Sebastien Dery

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from src.io.utils import load_categoryfile
from src.io.batch import fetch_naturetitles


"""
-------------------------------------------------------------------------------
                    Loading publication dataset for categories
-------------------------------------------------------------------------------
"""


categories = [
    'neuroscience',
    'biochemistry',
    'astrophysics'
]
categories = None

print('_' * 80)
print("Loading Nature dataset for categories")
print(categories if categories else "all")

data_train = fetch_naturetitles(subset = 'train', 
                                categories = categories, 
                                shuffle = True)

data_test = fetch_naturetitles(subset = 'test', 
                               categories = categories, 
                               shuffle = True)

print('Data loaded')

# Reload for case categories == None
categories = data_train['target_names']

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train['data'])
data_test_size_mb = size_mb(data_test['data'])

print("%d documents - %0.3fMB (training set)" % (
    len(data_train['data']), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test['data']), data_test_size_mb))
print("%d categories" % len(categories))

# Split a training set and a test set
y_train, y_test = data_train['target'], data_test['target']






"""
-------------------------------------------------------------------------------
    Extracting features from the training data using a sparse vectorizer
-------------------------------------------------------------------------------
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from AbstractVectorizer import AbstractVectorizer

from time import time
import numpy as np

print('_' * 80)
print("Extracting features from the training data using a sparse vectorizer")
t0 = time()

n_features = 1000

use = 'hashing'
#use = 'doc2vec'
#use = 'tfidf'
if use == 'hashing':
    vectorizer = HashingVectorizer(stop_words='english', n_features=n_features, non_negative=True)
    X_train = vectorizer.transform(data_train['data'])
elif use=='doc2vec':
    vectorizer = AbstractVectorizer(n_features=n_features, alpha=0.001, window=1)
    X_train = vectorizer.fit_transform(data_train['data'])
else:
    # sublinear_tf: replace tf with 1 + log(tf)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=n_features, sublinear_tf=True, max_df=0.5)
    X_train = vectorizer.fit_transform(data_train['data'])
    
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)






"""
-------------------------------------------------------------------------------
    Extracting features from the training data using a sparse vectorizer
-------------------------------------------------------------------------------
"""

print('_' * 80)
print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test['data'])
    
#X_test = np.array(model.docvecs)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)






"""
-------------------------------------------------------------------------------
                            Standardizing Benchmark
-------------------------------------------------------------------------------
"""

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics

from src.analysis.benchmark import benchmark

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf,X_train,y_train,X_test,y_test))


for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3),X_train,y_train,X_test,y_test))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty),X_train,y_train,X_test,y_test))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet"),X_train,y_train,X_test,y_test))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid(),X_train,y_train,X_test,y_test))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
]),X_train,y_train,X_test,y_test))

"""
# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
"""