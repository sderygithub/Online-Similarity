"""
Demo
Interactive classification fo user-generated sentences

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

"""
-------------------------------------------------------------------------------
                                Interactive fun
-------------------------------------------------------------------------------
"""

import os
from src.io.utils import load_sys
from src.io.utils import fetch_category_lookup

print('-' * 80)
print("Welcome to Interactive Classification Demo")
print(" ")
print("""You will shortly be asked to provide sentences which will then be
classified based on the input trained classifier. Don't worry nothing critical 
is going to happen. This is purely to have some fun and see how various 
permutation of sentences are classified.
Example: Internally generated cell assembly sequences in the rat hippocampus""")

# Category lookup table to translate results
category_id_lookup = fetch_category_lookup()

# Load saved classifiers
cls_path = os.path.normpath(os.path.join(os.getcwd(),'classifiers','nb_multinomial.pickle'))
vec, clf = load_sys(cls_path)

# Until user gets bored
while True:
    print '-' * 80
    i = raw_input("Enter Scientific sounding sentence (or Enter to quit):\n")
    if not i:
        break
    X = vec.transform([i])
    y = clf.predict(X)
    print category_id_lookup[y[0]].upper()

