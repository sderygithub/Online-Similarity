"""
-------------------------------------------------------------------------------
                                    Batch 
-------------------------------------------------------------------------------
"""

import os
from src.io.utils import fetch_categories
from src.io.utils import load_categoryfile

def fetch_naturetitles(subset, categories, shuffle):
    """
    Load sentences within a file
    """
    datafolder = "/Users/sdery/Desktop/Projects/Semantic_Extraction/yewno/data"
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