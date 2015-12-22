"""
-------------------------------------------------------------------------------
                                    Streaming
-------------------------------------------------------------------------------
"""

import numpy as np
import itertools
import os.path
import os

from utils import load_categoryfile


def get_minibatch(doc_iter, size):
    """
    Extract a minibatch of examples, return a tuple X_text, y.
    Note: size is before excluding invalid docs with no topics assigned.

    """
    data = [(doc[0], doc[1]) for doc in itertools.islice(doc_iter, size)]
    if not len(data):
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    X_text, y = zip(*data)
    return X_text, np.asarray(y, dtype=int)



def iter_minibatches(doc_iter, minibatch_size):
    """
    Generator of minibatches.

    """
    X_text, y = get_minibatch(doc_iter, minibatch_size)
    while len(X_text):
        yield X_text, y
        X_text, y = get_minibatch(doc_iter, minibatch_size)



def stream_nature_documents(data_path=None,subset=None,categories=None):
    """Iterate over documents of the Nature dataset.

    The Reuters archive will automatically be downloaded and uncompressed if
    the `data_path` directory does not exist.

    Documents are represented as dictionaries with 'body' (str),
    'title' (str), 'topics' (list(str)) keys.

    """
    if subset is None:
        subset = 'train'

    datafolder = "/Users/sdery/Desktop/Projects/Semantic_Extraction/yewno/data"
    
    # Build category numerical id lookup table
    category_path = "/Users/sdery/Desktop/Projects/Semantic_Extraction/yewno/categories.txt"
    category_id_lookup = {}
    with open(category_path,'r') as f:
        for uid,row in enumerate(f):
            category_id_lookup[row.split(',')[0].lower().strip()] = uid

    # If none specified, assume all
    if categories is None:
        categories = category_id_lookup.keys()

    # Lazy stream back sequentially per category
    for uid,category_name in enumerate(categories):
        c_file = "%s/%s_%s.txt" % (datafolder,category_name,subset)
        data = load_categoryfile(c_file)
        for row in data:
            yield (row, category_id_lookup[category_name], category_name)

