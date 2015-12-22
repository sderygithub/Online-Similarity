"""
Demo 
Online Naive Bayesian Learning for Publication Classification

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


from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer

import os
import numpy as np
from time import time

from src.analysis.feature_extraction.text import AbstractVectorizer
from src.io.streaming import *
from src.io.utils import save_sys


print('_' * 80)
print("Welcome to Online Learning Demo")



"""
-------------------------------------------------------------------------------
                                 Test conditions
-------------------------------------------------------------------------------
"""

# Which categories should we test on
categories = [
    'microbiology',
    'biochemistry',
    'astrophysics',
    'botany',
    'climatology',
    'epidemiology',
    'geophysics',
    'neuroscience',
    'quantum'
]

# Use all categories
#categories = None


# Hardcoded number of features for the sparse vectorizer
n_features = 2000


# We will feed the classifier with mini-batches of 1000 documents; this means
# we have at most 1000 docs in memory at any time.  The smaller the document
# batch, the bigger the relative overhead of the partial fit methods.
minibatch_size = 100


# This work laid the foundation for comparing multiple vectorizer
# At this point in time, the Doc2Vec vectorizer is inadequate
vectorizer_type = 'hashing'


print("On the menu today")
print("Categories: %s" % (','.join(categories)))
print("Number of features: %i" % (n_features))
print("Minibatch size: %i" % (minibatch_size))
print("Vectorizer: %s" % (vectorizer_type))
print('_' * 80)


"""
-------------------------------------------------------------------------------
                            Initializing vectorizer
-------------------------------------------------------------------------------
"""


print("Initializing %s vectorizer" % vectorizer_type)

#use = 'doc2vec'
tick = time()

if vectorizer_type == 'hashing':
    # Create the vectorizer and limit the number of features to a reasonable
    # maximum
    vectorizer = HashingVectorizer(stop_words='english', 
                                    n_features=n_features, 
                                    non_negative=True)
elif vectorizer_type == 'doc2vec':
    # Still in prototype phase
    vectorizer = AbstractVectorizer(n_features=n_features,
                                     alpha=0.001, 
                                     window=4)

initializing_vectorizer_time = time() - tick




"""
-------------------------------------------------------------------------------
                    Loading test dataset for categories
-------------------------------------------------------------------------------
"""

# Iterator over parsed Nature files dedicated for testing.
test_stream = stream_nature_documents(data_path=None,subset='test',categories=categories)

# First we hold out a number of examples to estimate accuracy
n_test_documents = 10000

print("Parsing a maximum of %i documents from the testing repository" % n_test_documents)

tick = time()
X_test_text, y_test = get_minibatch(test_stream, n_test_documents)
parsing_time = time() - tick

print("Number of documents: %i" % (len(y_test)))
print("Number of classes: %i" % (len(np.unique(y_test))))

print("Extracting features from the testing dataset using a sparse vectorizer")

tick = time()
X_test = vectorizer.transform(X_test_text)
vectorizing_time = time() - tick






"""
-------------------------------------------------------------------------------
    Extracting features from the training data using a sparse vectorizer
-------------------------------------------------------------------------------
"""


def progress(cls_name, stats):
    """
    Report progress information, return a string.

    """
    duration = time() - stats['t0']
    s = "%20s classifier : \t" % cls_name
    s += "accuracy: %(accuracy).3f " % stats
    s += "in %.2fs (%5d docs/s)" % (duration, stats['n_train'] / duration)
    return s


# In this experiment we'll try a few different classifiers to see how well
# they port themselves to online textual learning.
partial_fit_classifiers = {
    'SGD': SGDClassifier(),
    'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier(),
}

# Output statistics structure
cls_stats = {}
for cls_name in partial_fit_classifiers:
    stats = {'n_train': 0, 'n_train_pos': 0,
             'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time(),
             'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
    cls_stats[cls_name] = stats



# Iterator over parsed Nature files dedicated for testing.
data_stream = stream_nature_documents(data_path=None,subset='train',categories=categories)


# Multiclass classification problem
all_classes = np.unique(y_test)


# Create the data_stream that parses Reuters SGML files and iterates on
# documents as a stream.
minibatch_iterators = iter_minibatches(data_stream, minibatch_size)


# Let's keep track of how much time is spent on vectorizing
total_vect_time = 0.0


print("Main learning loop")
print(" ")

#  Main loop: iterate on mini-batchs of examples
for i, (X_train_text, y_train) in enumerate(minibatch_iterators):
    
    # Vectorize new training set
    tick = time()
    X_train = vectorizer.transform(X_train_text)
    total_vect_time += time() - tick

    # For each classifier
    for cls_name, cls in partial_fit_classifiers.items():
        tick = time()
        
        # Update estimator with examples in the current mini-batch
        cls.partial_fit(X_train, y_train, classes=all_classes)

        # Accumulate test accuracy stats
        cls_stats[cls_name]['total_fit_time'] += time() - tick
        cls_stats[cls_name]['n_train'] += X_train.shape[0]
        cls_stats[cls_name]['n_train_pos'] += sum(y_train)
        
        tick = time()
        cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
        cls_stats[cls_name]['prediction_time'] = time() - tick

        # Accumulated history
        acc_history = (cls_stats[cls_name]['accuracy'], cls_stats[cls_name]['n_train'])
        cls_stats[cls_name]['accuracy_history'].append(acc_history)
        
        # Running history
        run_history = (cls_stats[cls_name]['accuracy'],
                       total_vect_time + cls_stats[cls_name]['total_fit_time'])
        cls_stats[cls_name]['runtime_history'].append(run_history)

        print(progress(cls_name, cls_stats[cls_name]))

    print('\n')






"""
-------------------------------------------------------------------------------
                                Save classifier
-------------------------------------------------------------------------------
"""


cls_path = os.path.normpath(os.path.join(os.getcwd(),'classifiers','naivebayes.pickle'))
save_sys(vectorizer,partial_fit_classifiers['NB Multinomial'],cls_path)


