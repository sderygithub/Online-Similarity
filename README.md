## Online-Learning ##
- by Sebastien Dery

### Welcome to Classifying the Branch of Science (CBoS) ###

This small project aims to lay the foundation for a streaming (mini-batch)
classification pipeline of Scientific publication according to their
location in the poverbial tree of knowledge (e.g. geography, biochemistry, 
particle physics, microbiology, neuroscience, etc.).

The challenge consist in having only a small subset of already labelled
documents which must be used during classification. Additional labels are
available but at a virtual cost.

##### Development timeline #####
	Day 1 
	  - Data extraction
	  - Sanity check and data exploration
	  - Ground-truths and labels
	  - Batch pipeline algorithms
	Day 2
	  - Mini-batch / Streaming pipeline algorithms
	  - Interactive Q/A learning scenario
	  - Validating algorithm and accuracy

Although very relevant in many applications, TF-IDF is not suitable for online
learning as the IDF of all past features will change with every new document -
which would mean re-visiting and re-training on all the previous documents 
(no-longer online). A straighforward alternative is to use Hashing of our paragraph as vectorizer which is then fed into classification algorithms. 



I also lay the foundation for an Paragraph Vectorizer which relies on the online
update of vocabulary for NN word vectorizer approach. At this moment in time it
only allows weight updates based on new training data. 



#### How would you build a classifier that improves its accuracy by minimizing the overall cost? ####

The problem at heart resides in taking maximum advantage of labelled data. 
A first stop would be to look for similarities between labelled and unlabelled
data to propagate the known label to documents where we have high confidence of
similarity. This work takes advantage of an NN vectorizer as formalized in
(Quoc Le & Tomas Mikolov) and implemented through Doc2Vec. The architecture
proposed two learning approach, “distributed memory” (dm) and “distributed 
bag of words” (dbow), with the former showing better results. 

Quoc Le & Tomáš Mikolov: “Distributed Representations of Sentences and Documents”
https://radimrehurek.com/gensim/models/doc2vec.html



### How would you assess the performances of your system? ###

##### Processing time #####

The performance of such a system can be evaluated at multiple scale. One would
need to monitor training time and the potential bottlenecks it includes 
(vectorizing? training classifier? increasing vocabulary?). 


##### Memory usage #####

When it comes to scalability, memory is often critical. Monitoring how much 
RAM is necessary and particularly in what format (sparse versus dense matrix,
 )


##### I/O #####

In any distributed system, input and output are often good starting point in 
terms of optimizing operation. The location of the data on the cluster may
need tweaking in order to have frequently used data in the same disk-space
so as to limit the overhead of communication.


##### Accuracy #####

As a classification problem the accuracy is also of great importance. Data
presented to the user should be relevant and useful. 


### If you were assigned additional experts, how would your strategy be affected? ###

With additional experts one must be careful not to overfit each personalities 
by having them label a specific subset of the data. Ideally, each should be
assigned various corners of the 


Variability within and across expert is also important to consider, as there 
is some level of subjectiveiy. Monitoring those variables by introducing
redundant data for them to label at different time point might be relevant 
assuming the budget allows it. With careful planning, this should point out 
what is intrinsicately ambigious about the data and what needs to be refined 
for the greatest accuracy. 


### Is your system scalable w.r.t. size of your dataset? ###

With the current implementation, all label vectors are stored separately in 
RAM. In the case above with a unique label per sentence, this causes memory 
usage to grow linearly with the size of the corpus. The similarity model to
propagate known labels would thus scale linearly. 




### Components ###
_______________________________________________________________________________

CBoS consists of:

     1) Python scripts.  These do not need to be compiled.
     2) Output logs. 	 



### Building ###
_______________________________________________________________________________

This project is build around concepts of functionality. No state are kept in
memory and no compiling is required



### Dependencies ###
_______________________________________________________________________________

pip install numpy
pip install sklearn
pip install gensim



### Installation ###
_______________________________________________________________________________



### Improvements ###
_______________________________________________________________________________

Dynamic Vocabulary

Adding new vocabulary to Word2Vec Neural Network is currently in discussion
and preliminary code has been pushed. I would pursue adding functionality to
Doc2Vec in similar fashion as to allow the online expansion of the vocabulary
for computing new weights. 
https://groups.google.com/forum/#!topic/gensim/UZDkfKwe9VI
http://rutumulkar.com/blog/2015/word2vec/

Note that this could also be done in batch mode while the streaming classification 
relies on previously computed vocabulary while still updating the weights


Dissimilarity

Assuming we are to ask an expert to label data, we should ask him to label
documents which are as dissimilar as possible in order to maximize the 
representation in feature space. The intuition is that having him label
a cluster of very simmilar item will not provide us with great coverage of 
the potential data that we ultimately need to cover.


Drifting trends

It is not uncommon in online learning that additional data becomes problematic.
This may stem from multiple factors like shifts in weight due to redundant 
data, class imbalance, etc. It is thus often useful to monitor trending in 
accuracy as the model evolves and compute offline model to take over when 
degradation goes below a certain threshold


Internationalization support

While NLP approaches are often constrained to english grammar, the world is 
filled with content of difference languages holding valuable information. 
Support for multiple language is a problem faced by many high-tech companies
looking to expand beyond the english-speaking world/web.


Multiple label support

The learning architecture of distributed neural network paragraph representation permits more than one label per sentence. While this work focused on a single label as proof of concept, multiple label is likely to be relevant. Embedding the intrinsic hierarchy of scientific ontology is likely to be benificial also.



Authors & Acknowledgements
_______________________________________________________________________________

Sebastien Dery



Copyright Notice and Disclaimer
_______________________________________________________________________________


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