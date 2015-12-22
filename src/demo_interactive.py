"""
-------------------------------------------------------------------------------
                                Interactive fun
-------------------------------------------------------------------------------
"""

import os

# Build category numerical id lookup table
category_path = os.path.normpath(os.path.join(os.getcwd(), 'categories.txt'))
category_id_lookup = []
with open(category_path,'r') as f:
    for uid,row in enumerate(f):
        category_id_lookup.append(row.split(',')[0].lower().strip())

# 
cls_path = os.path.normpath(os.path.join(os.getcwd(),'classifiers','nb_multinomial.pickle'))
vec, clf = load_clf(cls_path)

while True:
    print '-' * 80
    i = raw_input("Enter Scientific sounding sentence (or Enter to quit): ")
    if not i:
        break
    X = vec.transform([i])
    y = clf.predict(X)
    print category_id_lookup[y].upper()