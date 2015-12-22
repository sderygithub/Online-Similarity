"""
-------------------------------------------------------------------------------
                            Standardizing Benchmark
-------------------------------------------------------------------------------
"""

from sklearn import metrics
from time import time

def benchmark(clf,X_train,y_train,X_test,y_test):
    """
    Loading the keywords associated to a particular category
    Used in establishing some ground-truth of the data
    """

    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    
    clf_descr = str(clf).split('(')[0]

    return clf_descr, score, train_time, test_time