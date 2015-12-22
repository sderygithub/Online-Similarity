# Shuffle
c = list(zip(X_potential_text, y_train_truth))
shuffle(c)
shuffled_text, shuffled_truth = zip(*c)

# Build similarity model from new batch
similarity = AbstractVectorizer(n_features = n_features, alpha = 0.025, window = 5)
similarity.fit_transform(shuffled_text)

# Classifiers that support the 'partial_fit' method for online learning
partial_fit_classifiers = {
    'NaiveBayes':MultinomialNB(alpha=0.01),
    'NaiveBayes_Propagate':MultinomialNB(alpha=0.01)
}
# execfile('interactive_learning_study.py')

# Statistics structure
cls_stats = {}
cls_train = {}
for cls_name in partial_fit_classifiers:
    stats = {'n_train': 0, 'n_train_pos': 0,
             'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time(),
             'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
    cls_stats[cls_name] = stats
    data = {'data':[],'target':[],'weight':[]}
    cls_train[cls_name] = data

counter = 0
unvisited = range(len(shuffled_text))
outside_range = 500
unvisited = range(outside_range)
while unvisited:

    # Force feed ground truth
    target = shuffled_truth[unvisited[0]]

    # Compute similarity with other documents in batch
    doc = re.sub('[^a-zA-Z]', ' ', shuffled_text[unvisited[0]]).lower().split()
    new_doc_vec = similarity.infer_vector(doc)
    sims = similarity.docvecs.most_similar([new_doc_vec],topn=25)

    # Weighted labels
    indexes = [(unvisited[0], 1)]
    for ind,sim in enumerate(sims):
        if sim[1] > 0.6 and sim[0] > outside_range:
            indexes.append(sim)

    # Remove from navigation list
    for uid in [indexes[0]]:
        if uid[0] in unvisited:
            unvisited.remove(uid[0])

    # Append to training batch
    cls_train['NaiveBayes']['data'] = [shuffled_text[indexes[0][0]]]
    cls_train['NaiveBayes']['target'] = [target]
    cls_train['NaiveBayes']['weight'] = [indexes[0][1]]

    # List
    cls_train['NaiveBayes_Propagate']['data'] = [shuffled_text[ind[0]] for ind in indexes]
    cls_train['NaiveBayes_Propagate']['target'] = [target] * len(indexes)
    cls_train['NaiveBayes_Propagate']['weight'] = [ind[1] for ind in indexes]

    # For each classifier
    for cls_name, cls in partial_fit_classifiers.items():
        tick = time()
        
        # Vectorize new training set
        X_train = vectorizer.transform(cls_train[cls_name]['data'])
        y_train = cls_train[cls_name]['target']
        sample_weight = cls_train[cls_name]['weight']
        
        # Update estimator with examples in the current mini-batch
        cls.partial_fit(X_train, y_train, classes=all_classes)

        # Accumulate test accuracy stats
        cls_stats[cls_name]['total_fit_time'] += time() - tick
        cls_stats[cls_name]['n_train'] += X_train.shape[0]
        cls_stats[cls_name]['n_train_pos'] += sum(y_train)
        
        tick = time()
        cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
        cls_stats[cls_name]['prediction_time'] = time() - tick

        # Accumulated history
        acc_history = (cls_stats[cls_name]['accuracy'], cls_stats[cls_name]['n_train'])
        cls_stats[cls_name]['accuracy_history'].append(acc_history)
        
        # Running history
        run_history = (cls_stats[cls_name]['accuracy'],
                       cls_stats[cls_name]['total_fit_time'])
        cls_stats[cls_name]['runtime_history'].append(run_history)

        print(progress(cls_name, cls_stats[cls_name]))

        print('\n')

 

import pylab as plt

def google_color(n):
  colores_g = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#0099c6", "#dd4477", "#66aa00", "#b82e2e", "#316395", "#994499", "#22aa99", "#aaaa11", "#6633cc", "#e67300", "#8b0707", "#651067", "#329262", "#5574a6", "#3b3eac"];
  return colores_g[n % len(colores_g)];

import numpy as np
def smooth(x,window_len=5,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:  
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

figure_fontsize = 20
cid = 0
for cls_name, cls in partial_fit_classifiers.items():
    y = [c[0] for c in cls_stats[cls_name]['runtime_history']]
    x = range(len(y))
    yy = smooth(np.array(y))
    area = np.pi * (6 * np.array([1] * len(x))) * 2.0
    #plt.scatter(x, y, s = area, facecolor = google_color(cid), edgecolor = 'white', alpha = 0.8)
    plt.plot(x, yy, '-', color = google_color(cid), linewidth = 3.0, alpha=0.6)
    plt.grid()
    plt.title('', size = figure_fontsize)
    plt.xlabel('Mini-Batch', size = figure_fontsize)
    plt.ylabel('Accuracy', size = figure_fontsize)
    plt.title('Propagate Label (Blue) versus Baseline (Red)')
    plt.ylim([0, 1.05])
    plt.xlim([x[0], x[-1]])
    cid += 1

#plt.show()

fig = plt.gcf()
fig.set_size_inches(12, 6)
fig.savefig('OnlineLearningNaiveBayes_Propagate_3.png', dpi=300)
plt.clf()

   