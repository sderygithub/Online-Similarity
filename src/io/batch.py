"""
Batch I/O utility functions

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

import os
from src.io.utils import fetch_categories
from src.io.utils import load_categoryfile

def fetch_naturetitles(subset, categories, shuffle):
    """
    Load sentences within a file
    """
    datafolder = "/data"
    if categories is None:
        categories = fetch_categories()

    # Basic structure
    dataset = {'desc':'', 'data':[], 'target':[], 'target_names':[]}

    for uid,category_name in enumerate(categories):
        c_file = "%s/%s_%s.txt" % (datafolder,category_name,subset)
        data = load_categoryfile(c_file)
        dataset['data'].extend(data)
        dataset['target_names'].append(category_name)
        dataset['target'].extend([uid] * len(data))

    return dataset
