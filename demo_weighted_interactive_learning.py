"""
Demo 
Interactive Learning

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
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer

from src.analysis.feature_extraction.text import AbstractVectorizer
from src.io.streaming import stream_nature_documents
from src.io.streaming import iter_minibatches
from src.io.streaming import get_minibatch

from time import time
import numpy as np



print('_' * 80)
print("Welcome to Interactive Learning Demo")
print(" ")
print("On the menu today")
print("""Starting from a cold-start (no training data), let's build a classifier
through question and answering scenario with the user. This program will first start
by vectorizing the test dataset which we will use to keep track of accuracy as the
user provides more sample. For each new batch data, a similarity model is built
between paragraphs. This is used to propagate user-provided labels and reduce
the overall amount of question required for reasonable accuracy.
""")
print('-' * 80)

"""
-------------------------------------------------------------------------------
                                 Test conditions
-------------------------------------------------------------------------------
"""


# Which categories should we test on
categories = [
    'microbiology',
    'astrophysics',
    'climatology',
    'geophysics',
    'neuroscience',
    'quantum'
]

# Build category numerical id lookup table
category_path = "/Users/sdery/Desktop/Projects/Semantic_Extraction/yewno/categories.txt"
category_id_lookup = {}
with open(category_path,'r') as f:
    for uid,row in enumerate(f):
        category_id_lookup[row.split(',')[0].lower().strip()] = uid

# For interactive display
category_display = ','.join(categories)


#categories = None

# Hardcoded number of features for the sparse vectorizer
n_features = 2000


# We chose a high value here because data are stored sequentially. This gets
# a little boring for the user as its mostly the same kind of topic. By forcing
# a large value, ww can shuffle the data and have more diversity.
minibatch_size = 10000


# This work laid the foundation for comparing multiple vectorizer
# At this point in time, the Doc2Vec vectorizer is inadequate
vectorizer_type = 'hashing'



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

print("Parsing %i documents from the testing data repository" % n_test_documents)

tick = time()
X_test_text, y_test = get_minibatch(test_stream, n_test_documents)
parsing_time = time() - tick

print("  Number of documents: %i" % (len(y_test)))
print("  Number of classes: %i" % (len(np.unique(y_test))))

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
partial_fit_classifiers = {'NaiveBayes':MultinomialNB(alpha=0.01)}

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


# Create the data_stream that trickles publication files
minibatch_iterators = iter_minibatches(data_stream, minibatch_size)


import re
from random import shuffle


#  Main loop: iterate on mini-batchs of examples
for i, (X_potential_text, y_train_truth) in enumerate(minibatch_iterators):
    
    print("Receiving new batch of data")
    break;
    # Necessary for delete operation
    X_potential_text = list(X_potential_text)
    y_train_truth = list(y_train_truth)

    # Shuffle?
    c = list(zip(X_potential_text, y_train_truth))
    shuffle(c)
    shuffled_text, shuffled_truth = zip(*c)


    # Build similarity model from new batch
    print("Building similarity model (might take a while if using large mini-batch size)")
    print("  (e.g. 273.18s for 7950 on a MacBook Pro 2012)")
    tick = time()
    similarity = AbstractVectorizer(n_features = n_features,
                                    alpha = 0.001, 
                                    window = 5)
    similarity.fit_transform(shuffled_text)
    similarity_model_time = time() - tick
    print("Done in %.2fs" % similarity_model_time)


    # Main loop
    X_train_text = []
    y_train = []
    y_weight = []

    unvisited = range(len(X_potential_text))
    while unvisited:

        print ' ' * 80
        print '-' * 80
        i = raw_input("""
            Please classify this paragraph according to one of the following labels (or Enter to quit)
            \n %s
            \n %s\n: """ % (category_display, X_potential_text[unvisited[0]]))

        if not i:
            break

        if i not in categories:
            print("Sorry, the label %s is currently unknown" % i)

        else:
            # Compute similarity with other documents in batch
            doc = re.sub('[^a-zA-Z]', ' ', X_potential_text[unvisited[0]]).lower().split()
            new_doc_vec = similarity.infer_vector(doc)
            sims = similarity.docvecs.most_similar([new_doc_vec])

            # Label all those above a certain threshold
            # The threshold is currently arbitrary and subject to improvement
            indexes = [(unvisited[0], 1.0)]
            for uid in range(10):
                if uid in unvisited:
                    indexes.append(sims[uid])

            # Remove from navigation list
            for uid in indexes:
                if uid[0] in unvisited:
                    unvisited.remove(uid[0])

            # Append to training batch
            X_train_text.extend([X_potential_text[tid[0]] for tid in indexes])
            y_train.extend([i] * len(indexes))
            y_weight.extend([tid[1] for tid in indexes])

            # Vectorize new training set
            X_train = vectorizer.transform(X_train_text)

            # For each classifier
            for cls_name, cls in partial_fit_classifiers.items():
                tick = time()
                
                # Update estimator with examples in the current mini-batch
                cls.partial_fit(X_train, y_train, classes=all_classes, sample_weight=y_weight)

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
                               cls_stats[cls_name]['total_fit_time'])
                cls_stats[cls_name]['runtime_history'].append(run_history)

                print(progress(cls_name, cls_stats[cls_name]))

            print('\n')

            # Reset training batch
            X_train_text = []
            y_train = []
            y_weight = []
           
           


#def main(argv):
#    opt = docopt(__doc__, argv)

#if __name__ == "__main__":
#    try:
#        main(sys.argv[1:])
#    except KeyboardInterrupt:
#        pass
