"""
-------------------------------------------------------------------------------
                                    Utils
-------------------------------------------------------------------------------
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




"""
def stream_nature_documents(data_path=None):
    ""Iterate over documents of the Nature dataset.

    The Reuters archive will automatically be downloaded and uncompressed if
    the `data_path` directory does not exist.

    Documents are represented as dictionaries with 'body' (str),
    'title' (str), 'topics' (list(str)) keys.

    ""

    DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'reuters21578-mld/reuters21578.tar.gz')
    ARCHIVE_FILENAME = 'reuters21578.tar.gz'

    if data_path is None:
        data_path = os.path.join(get_data_home(), "reuters")
    if not os.path.exists(data_path):
        # Download the dataset.
        print("downloading dataset (once and for all) into %s" %
              data_path)
        os.mkdir(data_path)

        def progress(blocknum, bs, size):
            total_sz_mb = '%.2f MB' % (size / 1e6)
            current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
            if _not_in_sphinx():
                print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb),
                      end='')

        archive_path = os.path.join(data_path, ARCHIVE_FILENAME)
        urllib.request.urlretrieve(DOWNLOAD_URL, filename=archive_path,
                                   reporthook=progress)
        if _not_in_sphinx():
            print('\r', end='')
        print("untarring Reuters dataset...")
        tarfile.open(archive_path, 'r:gz').extractall(data_path)
        print("done.")

    for filename in glob(os.path.join(data_path, "*.sgm")):
        for doc in parser.parse(open(filename, 'rb')):
            yield doc
"""

"""
datafolder = "/Users/sdery/Desktop/Projects/Semantic_Extraction/yewno/data"
category_name = 'astrophysics'
c_file = "%s/%s.txt" % (datafolder,category_name)
data = load_categoryfile(c_file)

dataset = fetch_naturetitles(subset='train',categories=['astrophysics','biochemistry'],shuffle=False)
"""