"""
General I/O utility functions

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

from sets import Set
import os
import pickle


def load_categoryfile(c_file):
    """
    Simple wrapper for loading a text file
    """
    data = []
    with open(c_file,'r') as f:
        data = [d.strip() for d in f.readlines()]
    return data

def dataset2dict(ds):
    """
    Load sentences within a file
    """
    dd = {}
    for uid,row in enumerate(ds['target']):
        if ds['target_names'][row] in dd:
            dd[ds['target_names'][row]].add(ds['data'][uid])
        else:
            dd[ds['target_names'][row]] = Set([ds['data'][uid]])
    return dd

def merge_dataset(ds1,ds2):
    """
    Load sentences within a file
    """
    dataset = {'desc':'', 'data':[], 'target':[], 'target_names':[]}
    # 
    dd = dataset2dict(ds1)
    # 
    for uid,row in enumerate(ds2['target']):
        if ds2['target_names'][row] in dd:
            dd[ds2['target_names'][row]].add(ds2['data'][uid])
        else:
            dd[ds2['target_names'][row]] = Set([ds2['data'][uid]])
    #
    dataset['target_names'] = dd.keys()
    for uid,name in enumerate(dd.keys()):
        dataset['data'].extend(list(dd[name]))
        dataset['target'].extend([uid] * len(dataset['data']))
    
    return dataset


def fetch_categories():
    """
    Compile categories from file into a single list
    Used primarily has aid function when no categories are provided
    """
    # Build category numerical id lookup table
    category_path = os.path.normpath(os.path.join(os.getcwd(), 'categories.txt'))
    category_id_lookup = []
    with open(category_path,'r') as f:
        for uid,row in enumerate(f):
            category_id_lookup.append(row.split(',')[0].lower().strip())
    return category_id_lookup


def fetch_category_lookup():
    """
    Build lookup table to translate classification label into 
    human readable names
    """
    # Build category numerical id lookup table
    category_path = os.path.normpath(os.path.join(os.getcwd(), 'categories.txt'))
    category_id_lookup = {}
    with open(category_path,'r') as f:
        for uid,row in enumerate(f):
            category_id_lookup[uid] = row.split(',')[0].lower().strip()
    return category_id_lookup


def save_sys(vec,clf,file_path):
    with open(file_path, 'wb') as f:
        pickle.dump((vec,clf), f)

def load_sys(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_clf(clf,file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(clf, f)

def load_clf(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
