def normalize_and_rgb(images): 
    import numpy as np
    #normalize image to 0-255 per image.
    image_sum = 1/np.sum(np.sum(images,axis=1),axis=-1)
    given_axis = 0
    # Create an array which would be used to reshape 1D array, b to have 
    # singleton dimensions except for the given axis where we would put -1 
    # signifying to use the entire length of elements along that axis  
    dim_array = np.ones((1,images.ndim),int).ravel()
    dim_array[given_axis] = -1
    # Reshape b with dim_array and perform elementwise multiplication with 
    # broadcasting along the singleton dimensions for the final output
    image_sum_reshaped = image_sum.reshape(dim_array)
    images = images*image_sum_reshaped*255

    # make it rgb by duplicating 3 channels.
    images = np.stack([images, images, images],axis=-1)
    
    return images

def image_with_label(train_file, istart,iend):
    import tables
    import numpy as np
    f = tables.open_file(train_file, 'r')
    a = np.array(f.root.img_pt) # Images
    b = np.array(f.root.label) # Labels
    return normalize_and_rgb(a[istart:iend]),b[istart:iend]

def count_events(train_files):
    import tables
    n_events = 0
    for train_file in train_files:
        f = tables.open_file(train_file, 'r')
        n_events += f.root.label.shape[0]
    return n_events

# Create a heatmap of the training file hits.
# Useful for visually confirming the data is well-distributed.
def test_heatmap(train_files):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    a, b = image_with_label(train_files[0],0,20000)
    new_a = np.swapaxes(a[:,:,:,0],0,2)
    new_a = np.swapaxes(new_a,0,1)
    c = np.dot(new_a,b[:,0])
    d = np.dot(new_a,b[:,1])
#     %matplotlib inline
    #mpl.use('agg')

    width = 64
    height = 64
    fontsize = 120

    plt.figure(figsize=(width,height))
    ax = plt.subplot() 
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(fontsize)
    plt.imshow(c, norm=mpl.colors.LogNorm(), origin='lower', interpolation='nearest',label='top')
    cbar = plt.colorbar(shrink=0.82)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label(r'$p_T$', fontsize=fontsize)
    plt.xlabel(r'$i\eta$', fontsize=fontsize)
    plt.ylabel(r'$i\phi$', fontsize=fontsize)
    plt.savefig('top.pdf')

    plt.figure(figsize=(width,height))
    ax = plt.subplot() 
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(fontsize)
    plt.imshow(d, norm=mpl.colors.LogNorm(), origin='lower', interpolation='nearest',label='QCD')
    cbar = plt.colorbar(shrink=0.82)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label(r'$p_T$', fontsize=fontsize)
    plt.xlabel(r'$i\eta$', fontsize=fontsize)
    plt.ylabel(r'$i\phi$', fontsize=fontsize)
    plt.savefig('QCD.pdf')

def preprocess_images():
    import tensorflow as tf
    # Create a placeholder for our incoming images
    in_images = tf.placeholder(tf.float32)
    in_height = 64
    in_width = 64
    in_images.set_shape([None, in_height, in_width, 3])
    
    # Resize those images to fit our featurizer
    out_width = 224
    out_height = 224
    image_tensors = tf.image.resize_images(in_images, [out_height,out_width])
    image_tensors = tf.to_float(image_tensors)
    
    return in_images, image_tensors

def construct_classifier():
    from keras.layers import Dropout, Dense, Flatten, Input
    from keras.models import Model
    from keras import backend as K
    import tensorflow as tf
    K.set_session(tf.get_default_session())
    
    FC_SIZE = 1024
    NUM_CLASSES = 2

    in_layer = Input(shape=(1, 1, 2048,),name='input_1')
    x = Dropout(0.2, input_shape=(1, 1, 2048,),name='dropout_1')(in_layer)
    x = Dense(FC_SIZE, activation='relu', input_dim=(1, 1, 2048,),name='dense_1')(x)
    x = Flatten(name='flatten_1')(x)
    preds = Dense(NUM_CLASSES, activation='softmax', input_dim=FC_SIZE, name='classifier_output')(x)
    
    model = Model(inputs = in_layer, outputs = preds)
    
    return model

def construct_model(quantized, saved_model_dir = None, starting_weights_directory = None, is_frozen=False, is_training=True):
    from azureml.contrib.brainwave.models import Resnet50, QuantizedResnet50
    import tensorflow as tf
    from keras import backend as K
    
    # Convert images to 3D tensors [width,height,channel]
    in_images, image_tensors = preprocess_images()

    # Construct featurizer using quantized or unquantized ResNet50 model
    
    if not quantized:
        featurizer = Resnet50(saved_model_dir, is_frozen=is_frozen, custom_weights_directory = starting_weights_directory)
    else:
        featurizer = QuantizedResnet50(saved_model_dir, is_frozen=is_frozen, custom_weights_directory = starting_weights_directory)
    
    features = featurizer.import_graph_def(input_tensor=image_tensors, is_training=is_training)
    
    # Construct classifier
    with tf.name_scope('classifier'):
        classifier = construct_classifier()
        preds = classifier(features)
    
    # Initialize weights
    sess = tf.get_default_session()
    tf.global_variables_initializer().run()
    
    if not is_frozen:
        featurizer.restore_weights(sess)
    
    if starting_weights_directory is not None:
        print("loading classifier weights from", starting_weights_directory+'/class_weights.h5')
        classifier.load_weights(starting_weights_directory+'/class_weights.h5')
        
    return in_images, image_tensors, features, preds, featurizer, classifier 

def check_model(preds, features, in_images, train_files, classifier):
    import tensorflow as tf
    from keras import backend as K
    
    sess = tf.get_default_session()
    in_labels = tf.placeholder(tf.float32, shape=(None, 2))
    a, b = image_with_label(train_files[0],0,1)
    c = classifier.layers[-1].weights[0]
    d = classifier.layers[-1].weights[1]
    print(" image:    ", a)
    print(" label:    ", b)
    print(" features: ", sess.run(features, feed_dict={in_images: a,
                                   in_labels: b,
                                   K.learning_phase(): 0}))
    print(" weights:  ", sess.run(c))
    print(" biases:   ", sess.run(d))    
    print(" preds:    ", sess.run(preds, feed_dict={in_images: a,
                                   in_labels: b,
                                   K.learning_phase(): 0}))
    
def chunks(files, chunksize): 
    """Yield successive n-sized chunks from a and b.""" 
    import tables
    import numpy as np
    for train_file in files: 
        f = tables.open_file(train_file, 'r') 
        a = np.array(f.root.img_pt) # Images 
        b = np.array(f.root.label) # Labels 
        c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
        np.random.shuffle(c)
        test_images = c[:, :a.size//len(a)].reshape(a.shape)
        test_labels = c[:, a.size//len(a):].reshape(b.shape)

        for istart in range(0,test_images.shape[0],chunksize):  
            yield normalize_and_rgb(test_images[istart:istart+chunksize]),test_labels[istart:istart+chunksize], len(test_labels[istart:istart+chunksize])

def train_model(preds, in_images, train_files, val_files, is_retrain = False, train_epoch = 10, classifier=None, saver=None, checkpoint_path=None): 
    """ training model """ 
    import tensorflow as tf
    from keras import backend as K
    from keras.objectives import binary_crossentropy 
    from keras.metrics import categorical_accuracy 
    from tqdm import tqdm

    learning_rate = 0.001 if is_retrain else 0.01

    # Specify the loss function
    in_labels = tf.placeholder(tf.float32, shape=(None, 2))   
    with tf.name_scope('xent'):
        cross_entropy = tf.reduce_mean(binary_crossentropy(in_labels, preds))
        
    with tf.name_scope('train'):  
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        #momentum = 0.9
        #optimizer_def = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
        #optimizer = optimizer_def.minimize(cross_entropy)
        optimizer_def = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = optimizer_def.minimize(cross_entropy)

    with tf.name_scope('metrics'):
        accuracy = tf.reduce_mean(categorical_accuracy(in_labels, preds))
        auc = tf.metrics.auc(tf.cast(in_labels, tf.bool), preds)
    
    sess = tf.get_default_session()
    # to re-initialize all variables 
    #sess.run(tf.group(tf.local_variables_initializer(),tf.global_variables_initializer()))
    # to re-initialize just local variables + optimizer variables
    sess.run(tf.group(tf.local_variables_initializer(),tf.variables_initializer(optimizer_def.variables())))
    
    # Create a summary to monitor cross_entropy loss
    tf.summary.scalar("loss", cross_entropy)
    # Create a summary to monitor accuracy 
    tf.summary.scalar("accuracy", accuracy)
    # Create a summary to monitor auc 
    tf.summary.scalar("auc", auc[0])
    # create a summary to look at input images
    tf.summary.image("images", in_images, 3)
    
    # Create summaries to visualize weights
    #for var in tf.trainable_variables():
    #    tf.summary.histogram(var.name.replace(':','_'), var)
        
    #grads = tf.gradients(cross_entropy, tf.trainable_variables())
    #grads = list(zip(grads, tf.trainable_variables()))
    
    # Summarize all gradients
    #for grad, var in grads:
    #    tf.summary.histogram(var.name.replace(':','_') + '/gradient', grad)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    chunk_size = 64
    n_train_events = count_events(train_files)
    train_chunk_num = int(n_train_events / chunk_size)+1
    
    train_writer = tf.summary.FileWriter(checkpoint_path + '/logs/train', sess.graph)
    val_writer = tf.summary.FileWriter(checkpoint_path + '/logs/val', sess.graph)

    loss_over_epoch = []
    accuracy_over_epoch = []
    auc_over_epoch = []
    
    val_loss_over_epoch = []
    val_accuracy_over_epoch = []
    val_auc_over_epoch = []
    best_val_loss = 999999

    for epoch in range(train_epoch):
        avg_loss = 0
        avg_accuracy = 0
        avg_auc = 0
        preds_temp = []
        label_temp = []
        i = 0
        for img_chunk, label_chunk, real_chunk_size in tqdm(chunks(train_files, chunk_size),total=train_chunk_num):
            _, loss, summary, accuracy_result, auc_result = sess.run([optimizer, 
                                                                      cross_entropy, 
                                                                      merged_summary_op,
                                                                      accuracy, auc],
                            feed_dict={in_images: img_chunk,
                                       in_labels: label_chunk,
                                       K.learning_phase(): 1})
            avg_loss += loss * real_chunk_size / n_train_events
            avg_accuracy += accuracy_result * real_chunk_size / n_train_events
            avg_auc += auc_result[0] * real_chunk_size / n_train_events
            train_writer.add_summary(summary, epoch * train_chunk_num + i)
            i += 1
        
        print("Epoch:", (epoch + 1), "loss = ", "{:.3f}".format(avg_loss))
        print("Training Accuracy:", "{:.3f}".format(avg_accuracy), ", Area under ROC curve:", "{:.3f}".format(avg_auc))

        loss_over_epoch.append(avg_loss)
        accuracy_over_epoch.append(avg_accuracy)
        auc_over_epoch.append(avg_auc)

        n_val_events = count_events(val_files)
        val_chunk_num = int(n_val_events / chunk_size)+1
        
        avg_val_loss = 0
        avg_val_accuracy = 0
        avg_val_auc = 0
        i = 0
        for img_chunk, label_chunk, real_chunk_size in tqdm(chunks(val_files, chunk_size),total=val_chunk_num):
            val_loss, val_accuracy_result, val_auc_result, summary = sess.run([cross_entropy, accuracy, auc, merged_summary_op],
                                feed_dict={in_images: img_chunk,
                                           in_labels: label_chunk,
                                           K.learning_phase(): 0})
            avg_val_loss += val_loss * real_chunk_size / n_val_events
            avg_val_accuracy += val_accuracy_result * real_chunk_size / n_val_events
            avg_val_auc += val_auc_result[0] * real_chunk_size / n_val_events
            val_writer.add_summary(summary, epoch * val_chunk_num + i)
            i += 1
            
        print("Epoch:", (epoch + 1), "val_loss = ", "{:.3f}".format(avg_val_loss))
        print("Validation Accuracy:", "{:.3f}".format(avg_val_accuracy), ", Area under ROC curve:", "{:.3f}".format(avg_val_auc))
    
        val_loss_over_epoch.append(avg_val_loss)
        val_accuracy_over_epoch.append(avg_val_accuracy)
        val_auc_over_epoch.append(avg_val_auc)
        
        if saver is not None and checkpoint_path is not None and classifier is not None:
            saver.save(sess, checkpoint_path+'/resnet50_bw', write_meta_graph=False, global_step = epoch)
            saver.save(sess, checkpoint_path+'/resnet50_bw', write_meta_graph=False)
            classifier.save_weights(checkpoint_path+'/class_weights-%s.h5'%epoch)
            classifier.save(checkpoint_path+'/class_model-%s.h5'%epoch)
            classifier.save_weights(checkpoint_path+'/class_weights.h5')
            classifier.save(checkpoint_path+'/class_model.h5')
            if avg_val_loss < best_val_loss:
                print("new best model")
                best_val_loss = avg_val_loss
                saver.save(sess, checkpoint_path+'/resnet50_bw_best', write_meta_graph=False)
                classifier.save_weights(checkpoint_path+'/class_weights_best.h5')
                classifier.save(checkpoint_path+'/class_model_best.h5')
                
        
    return loss_over_epoch, accuracy_over_epoch, auc_over_epoch, val_loss_over_epoch, val_accuracy_over_epoch, val_auc_over_epoch

def test_model(preds, in_images, test_files):
    """Test the model"""
    import tensorflow as tf
    from keras import backend as K
    from keras.objectives import binary_crossentropy 
    import numpy as np
    from keras.metrics import categorical_accuracy
    from tqdm import tqdm
    
    in_labels = tf.placeholder(tf.float32, shape=(None, 2))
    
    cross_entropy = tf.reduce_mean(binary_crossentropy(in_labels, preds))
    accuracy = tf.reduce_mean(categorical_accuracy(in_labels, preds))
    auc = tf.metrics.auc(tf.cast(in_labels, tf.bool), preds)
   
    chunk_size = 64
    n_test_events = count_events(test_files)
    chunk_num = int(n_test_events/chunk_size)+1
    preds_all = []
    label_all = []
    
    sess = tf.get_default_session()
    sess.run(tf.local_variables_initializer())
    
    avg_accuracy = 0
    avg_auc = 0
    avg_test_loss = 0
    for img_chunk, label_chunk, real_chunk_size in tqdm(chunks(test_files, chunk_size),total=chunk_num):
        test_loss, accuracy_result, auc_result, preds_result = sess.run([cross_entropy, accuracy, auc, preds],
                        feed_dict={in_images: img_chunk,
                                   in_labels: label_chunk,
                                   K.learning_phase(): 0})
        avg_test_loss += test_loss * real_chunk_size / n_test_events
        avg_accuracy += accuracy_result * real_chunk_size / n_test_events
        avg_auc += auc_result[0]  * real_chunk_size / n_test_events 
        preds_all.extend(preds_result)
        label_all.extend(label_chunk)
    
    print("test_loss = ", "{:.3f}".format(avg_test_loss))
    print("Test Accuracy:", "{:.3f}".format(avg_accuracy), ", Area under ROC curve:", "{:.3f}".format(avg_auc))
    
    return avg_test_loss, avg_accuracy, avg_auc, np.asarray(preds_all).reshape(n_test_events,2), np.asarray(label_all).reshape(n_test_events,2)

# Save the results of the previous test.
# Provide the result saving directory, the prefix (see below), the label np array and the pred np array.
# It also expects a strict naming paradigm:
#   Non-quantized testing should be prefixed 't'
#   Quantized testing before fine-tuning should be prefixed 'q'
#   Quantized testing after fine-tuning should be prefixed 'ft'
#   Quantized testing on Brainwave should be prefixed 'b'
def save_results(results_dir, prefix, labels, preds):
    import numpy as np
    
    np.save(results_dir + "/" + prefix + "_labels.npy", labels)
    np.save(results_dir + "/" + prefix + "_preds.npy", preds)
    
# Once results have been compiled, use this function to plot them.
# It expects all the files to be there at runtime, so if they haven't yet been generated,
# comment out the relevant lines.
def plot_results(results_dir):
    import os
    import numpy as np
    from sklearn import metrics

    # Load the labels and results into memory.
    test_labels_t  = np.load(results_dir + "/t_labels.npy")
    test_preds_t   = np.load(results_dir + "/t_preds.npy")
#     test_labels_q  = np.load(results_dir + "/q_labels.npy")
#     test_preds_q   = np.load(results_dir + "/q_preds.npy")
#     test_labels_ft = np.load(results_dir + "/ft_labels.npy")
#     test_preds_ft  = np.load(results_dir + "/ft_preds.npy")
#     test_labels_b  = np.load(results_dir + "/s_labels.npy")
#     test_preds_b   = np.load(results_dir + "/s_preds.npy")
    
    # Determine the ROC curve for each of the tests. 
    # [:,0] will convert the labels from one-hot to binary.
    fpr_test_t, tpr_test_t, thresholds      = metrics.roc_curve(test_labels_t[:,0],  test_preds_t[:,0])
#     fpr_test_q, tpr_test_q, thresholds_q    = metrics.roc_curve(test_labels_q[:,0],  test_preds_q[:,0])
#     fpr_test_ft, tpr_test_ft, thresholds_ft = metrics.roc_curve(test_labels_ft[:,0], test_preds_ft[:,0])
#     fpr_test_b, tpr_test_b, thresholds_b    = metrics.roc_curve(test_labels_b[:,0],  test_preds_b[:,0])
    
    # Use the data we just generated to determine the area under the ROC curve.
    auc_test    = metrics.auc(fpr_test_t, tpr_test_t)
#     auc_test_q  = metrics.auc(fpr_test_q, tpr_test_q)
#     auc_test_ft = metrics.auc(fpr_test_ft, tpr_test_ft)
#     auc_test_b  = metrics.auc(fpr_test_b, tpr_test_b)

    # Plot the ROCs, labeling with the AUCs.
#     %matplotlib inline
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,5))
    plt.plot(tpr_test_t,    fpr_test_t,    label='Custom weights, AUC = %.2f%%'%(auc_test*100.))
#     plt.plot(tpr_test_q,  fpr_test_q,  label='Custom weights, quantized, AUC = %.2f%%'%(auc_test_q*100.))
#     plt.plot(tpr_test_ft, fpr_test_ft, label='Custom weights, quantized, fine-tuned, AUC = %.2f%%'%(auc_test_ft*100.))
#     plt.plot(tpr_test_b,  fpr_test_b,  label='Custom weights, Brainwave, AUC = %.2f%%'%(auc_test_b*100.))
    
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.ylim(0.00001,1)
    plt.xlim(0,1)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig('ROC_ft.pdf')

    # Find the true positive rate of 30% and 1 over the false positive rate at tpr = 30%.
    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx

    idx_t    = find_nearest(tpr_test_t,0.3)
#     idx_q  = find_nearest(tpr_test_q,0.3)
#     idx_ft = find_nearest(tpr_test_ft,0.3)
#     idx_b  = find_nearest(tpr_test_b,0.3)
    
    print (loss_t, accuracy_t, auc_test_t, tpr_test_t[idx_t], 1./fpr_test_t[idx_t])
#     print (loss_q, accuracy_q, auc_test_q, tpr_test_q[idx_q], 1./fpr_test_q[idx_q])
#     print (loss_ft, accuracy_ft, auc_test_ft, tpr_test_ft[idx_ft], 1./fpr_test_ft[idx_ft])
#     print (loss_b, accuracy_b, auc_test_b, tpr_test_b[idx_b], 1./fpr_test_b[idx_b])