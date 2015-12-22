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

# Create the vectorizer and limit the number of features to a reasonable
# maximum
vectorizer = HashingVectorizer(stop_words='english', 
                                n_features=n_features, 
                                non_negative=True)

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


# In this experiment we'll try a single classifiers with different 
# learning parameters 
partial_fit_classifiers = {
    'NB Multinomial 10': MultinomialNB(alpha=1.0),
    'NB Multinomial 09': MultinomialNB(alpha=0.9),
    'NB Multinomial 08': MultinomialNB(alpha=0.8),
    'NB Multinomial 07': MultinomialNB(alpha=0.7),
    'NB Multinomial 06': MultinomialNB(alpha=0.6),
    'NB Multinomial 05': MultinomialNB(alpha=0.5),
    'NB Multinomial 04': MultinomialNB(alpha=0.4),
    'NB Multinomial 03': MultinomialNB(alpha=0.3),
    'NB Multinomial 03': MultinomialNB(alpha=0.2),
    'NB Multinomial 01': MultinomialNB(alpha=0.1),
    'NB Multinomial 005': MultinomialNB(alpha=0.05),
    'NB Multinomial 004': MultinomialNB(alpha=0.04),
    'NB Multinomial 003': MultinomialNB(alpha=0.03),
    'NB Multinomial 002': MultinomialNB(alpha=0.02),
    'NB Multinomial 001': MultinomialNB(alpha=0.01),
    'NB Multinomial 0001': MultinomialNB(alpha=0.001)
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



"""
-------------------------------------------------------------------------------
                                Simple Plots
-------------------------------------------------------------------------------
"""


import pylab as plt

def google_color(n):
  colores_g = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#0099c6", "#dd4477", "#66aa00", "#b82e2e", "#316395", "#994499", "#22aa99", "#aaaa11", "#6633cc", "#e67300", "#8b0707", "#651067", "#329262", "#5574a6", "#3b3eac"];
  return colores_g[n % len(colores_g)];

import numpy as np
def smooth(x,window_len=5,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:  
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

figure_fontsize = 20

cid = 0
for cls_name, cls in partial_fit_classifiers.items():
    y = [c[0] for c in cls_stats[cls_name]['runtime_history']]
    x = range(len(y))
    yy = smooth(np.array(y))
    area = np.pi * (6 * np.array([1] * len(x))) * 2.0
    plt.scatter(x, y, s = area, facecolor = google_color(cid), edgecolor = 'white', alpha = 0.8)
    plt.plot(x, yy, '-', color = google_color(cid), linewidth = 2.0, alpha=0.6)
    plt.grid()
    plt.title('', size = figure_fontsize)
    plt.xlabel('Mini-Batch', size = figure_fontsize)
    plt.ylabel('Accuracy', size = figure_fontsize)
    plt.ylim([0, 1.05])
    plt.xlim([x[0], x[-1]])
    cid += 1

fig = plt.gcf()
fig.set_size_inches(12, 8)
fig.savefig('OnlineLearningNaiveBayes_MiniBatch.png', dpi=100)
plt.clf()


y = []
for cls_name, cls in partial_fit_classifiers.items():
    y.append(cls_stats[cls_name]['runtime_history'][-1][0])
x = range(len(y))
plt.scatter(x, y, s = area, facecolor = google_color(0), edgecolor = 'white', alpha = 0.8)
plt.plot(x, y, '-', color = google_color(0), linewidth = 2.0, alpha=0.6)
plt.title('', size = figure_fontsize)
plt.xlabel('Alpha', size = figure_fontsize)
plt.ylabel('Final Accuracy', size = figure_fontsize)
plt.xlim([x[0], x[-1]])

fig = plt.gcf()
fig.set_size_inches(12, 8)
fig.savefig('OnlineLearningNaiveBayes_Alpha.png', dpi=100)
plt.clf()


"""
-------------------------------------------------------------------------------
                                Save classifier
-------------------------------------------------------------------------------
"""


cls_path = os.path.normpath(os.path.join(os.getcwd(),'classifiers','naivebayes.pickle'))
save_sys(vectorizer,partial_fit_classifiers['NB Multinomial'],cls_path)


