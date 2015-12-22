"""
Demo 
Online Naive Bayesian Learning for Publication Classification

Usage:
    ./ipgeodecode.py ip_address
    ./ipgeodecode.py test
    ./ipgeodecode.py (-h | --help)

Options:
    -h, --help         Show this screen and exit.

Examples:

    ./

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

from analysis.AbstractVectorizer import AbstractVectorizer
from io.streaming import *
from io.utils import save_sys


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



"""
-------------------------------------------------------------------------------
                            Initializing vectorizer
-------------------------------------------------------------------------------
"""

print('_' * 80)
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

print('_' * 80)
print("Parsing %i documents from the testing data repository" % n_test_documents)

tick = time()
X_test_text, y_test = get_minibatch(test_stream, n_test_documents)
parsing_time = time() - tick

print("n_samples: %i, n_classes: %i" % (len(y_test), len(np.unique(y_test))))

print('_' * 80)
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

# Classifiers that support the 'partial_fit' method for online learning
partial_fit_classifiers = {
    'SGD': SGDClassifier(),
    'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier(),
}

# Statistics structure
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


cls_path = os.path.normpath(os.path.join(os.getcwd(),'classifiers','nb_multinomial.pickle'))
save_sys(vectorizer,partial_fit_classifiers['NB Multinomial'],cls_path)







import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot_accuracy(x, y, x_legend):
    # Plot accuracy as a function of x.
    x = np.array(x)
    y = np.array(y)
    plt.title('Classification accuracy as a function of %s' % x_legend)
    plt.xlabel('%s' % x_legend)
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(x, y)

rcParams['legend.fontsize'] = 10
cls_names = list(sorted(cls_stats.keys()))

# Plot accuracy evolution
plt.figure()
for _, stats in sorted(cls_stats.items()):
    # Plot accuracy evolution with #examples
    accuracy, n_examples = zip(*stats['accuracy_history'])
    plot_accuracy(n_examples, accuracy, "training examples (#)")
    ax = plt.gca()
    ax.set_ylim((0.8, 1))

plt.legend(cls_names, loc='best')

"""
plt.figure()
for _, stats in sorted(cls_stats.items()):
    # Plot accuracy evolution with runtime
    accuracy, runtime = zip(*stats['runtime_history'])
    plot_accuracy(runtime, accuracy, 'runtime (s)')
    ax = plt.gca()
    ax.set_ylim((0.8, 1))
plt.legend(cls_names, loc='best')
"""
"""
# Plot fitting times
plt.figure()
fig = plt.gcf()
cls_runtime = []
for cls_name, stats in sorted(cls_stats.items()):
    cls_runtime.append(stats['total_fit_time'])

cls_runtime.append(total_vect_time)
cls_names.append('Vectorization')
bar_colors = rcParams['axes.color_cycle'][:len(cls_names)]

ax = plt.subplot(111)
rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5,
                     color=bar_colors)

ax.set_xticks(np.linspace(0.25, len(cls_names) - 0.75, len(cls_names)))
ax.set_xticklabels(cls_names, fontsize=10)
ymax = max(cls_runtime) * 1.2
ax.set_ylim((0, ymax))
ax.set_ylabel('runtime (s)')
ax.set_title('Training Times')


def autolabel(rectangles):
    #attach some text vi autolabel on rectangles.
    for rect in rectangles:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                1.05 * height, '%.4f' % height,
                ha='center', va='bottom')

autolabel(rectangles)
plt.show()

# Plot prediction times
plt.figure()
#fig = plt.gcf()
cls_runtime = []
cls_names = list(sorted(cls_stats.keys()))
for cls_name, stats in sorted(cls_stats.items()):
    cls_runtime.append(stats['prediction_time'])
cls_runtime.append(parsing_time)
cls_names.append('Read/Parse\n+Feat.Extr.')
cls_runtime.append(vectorizing_time)
cls_names.append('Hashing\n+Vect.')
bar_colors = rcParams['axes.color_cycle'][:len(cls_names)]

ax = plt.subplot(111)
rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5,
                     color=bar_colors)

ax.set_xticks(np.linspace(0.25, len(cls_names) - 0.75, len(cls_names)))
ax.set_xticklabels(cls_names, fontsize=8)
plt.setp(plt.xticks()[1], rotation=30)
ymax = max(cls_runtime) * 1.2
ax.set_ylim((0, ymax))
ax.set_ylabel('runtime (s)')
ax.set_title('Prediction Times (%d instances)' % n_test_documents)
autolabel(rectangles)
plt.show()
"""
