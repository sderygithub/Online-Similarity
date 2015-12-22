X_train_text = []
y_train = []

# Necessary for delete operation
X_potential_text = list(X_potential_text)

# Shuffle?
shuffle(X_potential_text)

unvisited = range(len(X_potential_text))
while unvisited:

    print ' ' * 80
    print '-' * 80
    i = raw_input("""
        Please classify this paragraph according to one of the following labels (or Enter to quit)
        \n %s
        \n %s\n: """ % (category_display, X_potential_text[unvisited[0]]))

    if not i:
        break

    if i not in categories:
        print("Sorry, the label %s is currently unknown" % i)

    else:
        # Compute similarity with other documents in batch
        doc = re.sub('[^a-zA-Z]', ' ', X_potential_text[unvisited[0]]).lower().split()
        new_doc_vec = similarity.infer_vector(doc)
        sims = similarity.docvecs.most_similar([new_doc_vec])

        # Label all those above a certain threshold
        indexes = [unvisited[0]]
        for uid in range(10):
            if abs(sims[uid][1]) > 0.7 and uid in unvisited:
                indexes.append(int(sims[uid][0]))
        print("Found %i similar Abstracts" % (len(indexes) - 1))
        print(" ")

        # Remove from navigation list
        for uid in indexes:
            if uid in unvisited:
                unvisited.remove(uid)

        # Append to training batch
        X_train_text.extend([X_potential_text[tid] for tid in indexes])
        y_train.extend([category_id_lookup[i]] * len(indexes))

        # Vectorize new training set    
        X_train = vectorizer.transform(X_train_text)

        # For each classifier
        for cls_name, cls in partial_fit_classifiers.items():
            tick = time()
            
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

        # Reset training batch
        X_train_text = []
        y_train = []