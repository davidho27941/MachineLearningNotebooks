import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dropout, Dense, Flatten
import numpy as np
import tables
import glob
import random
from scipy.ndimage import zoom

base_model = ResNet50(weights='imagenet')

featurizer = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten_1').output)

FC_SIZE = 1024
NUM_CLASSES = 2
x = featurizer.output
x = Dropout(0.2)(x)
x = Dense(FC_SIZE, activation='relu')(x)
#x = Flatten()(x)
predictions = Dense(NUM_CLASSES, activation='softmax', name='classifier_output')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print(model.summary())


def normalize_and_rgb(images): 
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
    f = tables.open_file(train_file, 'r')
    a = np.array(f.root.img_pt) # Images
    b = np.array(f.root.label) # Labels
    return normalize_and_rgb(a[istart:iend]),b[istart:iend]


datadir = "../data/"
#num_train = 100  # Limit the number of images used in training to shorten epoch time

train_files = glob.glob(os.path.join(datadir, 'train_file_*'))

test_files = glob.glob(os.path.join(datadir, 'test/test_file_*'))


def chunks(train_files,chunksize):
    """Yield successive n-sized chunks from a and b."""
    for train_file in train_files:
        f = tables.open_file(train_file, 'r')
        a = np.array(f.root.img_pt) # Images
        b = np.array(f.root.label) # Labels
        for istart in range(0,a.shape[0],chunksize):
            yield zoom(normalize_and_rgb(a[istart:istart+chunksize]), (1, 3.5, 3.5, 1), order=1), b[istart:istart+chunksize]

n_events = 50000
chunksize = 64

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit_generator(chunks(train_files[:1], chunksize), steps_per_epoch = n_events/chunksize)


