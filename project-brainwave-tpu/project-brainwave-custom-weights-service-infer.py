#!/usr/bin/env python
# coding: utf-8
import os,sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np
from keras import backend as K
import tables
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

custom_weights_dir = os.path.expanduser("../weights-floatingpoint-224x224-fixval-best/")
custom_weights_dir_q = os.path.expanduser("../weights-quantized-224x224-fixval-best/")
saved_model_dir = os.path.expanduser("../machinelearningnotebooks/models/")
results_dir = os.path.expanduser("../results-quantized-224x224-fixval/")

from utils import normalize_and_rgb, image_with_label, count_events
import glob
datadir = "../../converted/rotation_224_v1/"
n_train_file = 122
n_test_file = 41
n_val_file = 41

train_files = glob.glob(os.path.join(datadir, 'train_file_*'))
test_files = glob.glob(os.path.join(datadir, 'test_file_*'))
val_files = glob.glob(os.path.join(datadir, 'val_file_*'))

n_train_events = count_events(train_files)
n_test_events = count_events(test_files)
n_val_events = count_events(val_files)

from utils import preprocess_images
from utils import construct_classifier

classifier = construct_classifier()
print("loading classifier weights from", custom_weights_dir_q+'/class_weights.h5')
classifier.load_weights(custom_weights_dir_q+'/class_weights.h5')

from azureml.core import Workspace

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
from azureml.core.webservice import Webservice
service_name = "modelbuild-service-8"
service = Webservice(ws, service_name)
print(service.ip_address + ':' + str(service.port))


from azureml.contrib.brainwave.client import PredictionClient
client = PredictionClient(service.ip_address, service.port)

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from tqdm import tqdm
from utils import chunks

chunk_size = 1  # Brainwave only processes one request at a time
if sys.argv[1]=='val':
    ifile = int(sys.argv[2])
    print('running on val data file %s'%ifile)
    files_to_run = val_files[ifile:ifile+1]
elif sys.argv[1]=='test':
    ifile = int(sys.argv[2])
    print('running on test data file %s'%ifile)
    files_to_run = test_files[ifile:ifile+1]


n_test_events = count_events(files_to_run)
chunk_num = int(n_test_events/chunk_size)+1

y_true = np.zeros((n_test_events,2))
y_feat = np.zeros((n_test_events,1,1,2048))
y_pred = np.zeros((n_test_events,2))

i = 0
for img_chunk, label_chunk, real_chunk_size in tqdm(chunks(files_to_run, chunk_size, max_q_size=1, shuffle=False), total=chunk_num):
    results = client.score_numpy_array(img_chunk)
    y_feat[i,:] = results
    y_pred[i,:] = classifier.predict(results.reshape(1,1,1,2048))[0,:]
    y_true[i,:] = label_chunk
    i+=1

from utils import save_results

accuracy = accuracy_score(y_true[:,0], y_pred[:,0]>0.5)
auc = roc_auc_score(y_true, y_pred)
save_results(results_dir, 'b_%s_%s'%(sys.argv[1],sys.argv[2]), accuracy, y_true, y_pred, y_feat)

print("Accuracy:", accuracy, "AUC:", auc)
