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

