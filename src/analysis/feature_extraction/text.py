"""
Online Paragraph Vectorizer
This is in development as update on the build_vocab function
is still unavailable. Will need to implement it

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

from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from numpy import array, zeros
import re

class AbstractVectorizer(Doc2Vec):
    """
    Loading the keywords associated to a particular category
    Used in establishing some ground-truth of the data
    """

    def __init__(self,alpha=0.025,n_features=300,window=5):
        Doc2Vec.__init__(self, size=n_features, alpha=alpha, min_alpha=alpha, window=window)
        self.n_features = n_features

    def fit_transform(self,sentences):
        sentences = list(LabeledSentence(re.sub('[^a-zA-Z]', ' ', value).lower().split(), ("ID_" + str(key),)) for key, value in enumerate(sentences))
        self.build_vocab(sentences)
        for epoch in range(10):
            self.train(sentences)
            # decrease the learning rate
            self.alpha /= 2
            # fix the learning rate, no decay
            self.min_alpha = self.alpha
        
    def transform(self,sentences):
        """
        Current implementation of infer_vector takes only a single row
        This wrapper simply iterates and fills an output matrix
        """
        matrix = zeros((len(sentences),self.n_features))
        for uid,row in enumerate(sentences):
            matrix[uid] = self.infer_vector(row)
        return matrix
    
    def update(self,sentences):
        # @Todo: Add vocabulary
        # self.build_vocab(sentences, update=True)
        self.train(sentences)

