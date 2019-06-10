import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
import tensorflow as tf
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from skimage.transform import resize
from skimage.morphology import label
#Keras tensor is a tensor object from the underlying backend(Theano, TensorFlow or CNTK)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
import cv2

#Fix images height, width and channels
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

#Define Data Paths
TRAIN_PATH = '/home/sumanthmeenan/Desktop/projects/U-Net/data/train/' 
TEST_PATH = '/home/sumanthmeenan/Desktop/projects/U-Net/data/test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

#os.walk(TRAIN_PATH) is a generator object
#next(os.walk(TRAIN_PATH))[1] returns list of objects in TRAIN_PATH
os.listdir(TRAIN_PATH) == next(os.walk(TRAIN_PATH))[1]

#Folders in Train and Test path
train_ids = next(os.walk(TRAIN_PATH))[1]
print('No. of examples in training data:', len(train_ids))
test_ids = os.listdir(TEST_PATH)
print('No. of examples in testing data:', len(test_ids))

#Analysis on 1 image
train_image = cv2.imread(TRAIN_PATH + train_ids[0] + '/images/' + os.listdir(TRAIN_PATH + train_ids[0] + '/images/')[0])
print('SHape of an image in training data',train_image.shape)
plt.imsave('/home/sumanthmeenan/j.png', train_image)

#Create Empty arrays 2 replace these later with resized images
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
print('shape of x_train is:', X_train.shape)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
print('shape of y_train is:', Y_train.shape)

#Getting and resizing train images and masks
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    #Training Images 
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    print('Before resizing image:',img.shape)
    
    #Different images are of different shape. lets resize them
    img1 = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    print('After resizing image:',img1.shape)
    
    #Replacing values in X_train
    X_train[n] = img1

    #Empty Mask (128, 128, 1)
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1))

    #Take all masks in folder and return 1 final mask for each input image
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        print('Before re-sizing mask', mask_.shape)
        mask1 = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True)
        print('After re-sizing mask', mask1.shape)
        
        mask1_ = np.expand_dims(mask1, axis=-1)
        print('After Expanding dimension of resized mask:', mask1_.shape)

        #for each element in 2 arrays, np.maximum() gives maximum value 
        mask = np.maximum(mask, mask1_)

    #Replacing Y_train values with full mask image 
    Y_train[n] = mask
    print('Y_train is', Y_train[n])

    #save the full mask (128,128)
    imsave(path + '/' + 'fullmask.png', np.resize(mask,(mask.shape[0],mask.shape[1])))



#Getting and resizing test images
sizes_test = []
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
print('shape of x_train is:', X_test.shape)

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    #Test Images 
    test_img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([test_img.shape[0], test_img.shape[1]])
    print('Before resizing image',test_img.shape)
    
    #Different images are of different shape. lets resize them
    test_img1 = resize(test_img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    print('After resizing image',test_img1.shape)
    
    #Replacing values in X_test
    X_test[n] = test_img1

"""X_train, Y_train, X_test tensors are updated"""

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
print('Shape of inputs: ', inputs.shape)
 
s = Lambda(lambda x: x/255) (inputs)
print('Shape of s: ', s.shape)

#1st block
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

#2nd block
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

#3rd block
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

#4th block
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

#5th block
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

#6th block
u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4]) #Its just Adding 2 arrays 
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

#7th block
u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

#8th block
u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

#9th block
u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

#Model compiling
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

earlystopper = EarlyStopping(patience=5, verbose=1) #stops if val.loss is same for 5 cont. epochs
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,callbacks=[earlystopper, checkpointer])

#Loading model weights, predicting for all train,test images
model = load_model('model-dsbowl2018-1.h5')
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
print('shape of preds_train is:', preds_train.shape)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
print('shape of preds_val is:', preds_val.shape)
preds_test = model.predict(X_test, verbose=1)
print('shape of preds_test is:', preds_test.shape)

#Applying Threshold for our predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

"""we stored test images original shapes, we resized test images shape and passed it into our model,
our output is (128,128,1), we are reshaping our mask image of test image into our original test image"""
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1]),mode='constant', preserve_range=True))

#saving predictions
ix = random.randint(0, len(preds_train_t))
imshow( X_train[ix])
imsave(X_train[ix])
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
imsave('/home/sumanthmeenan/masked.png', np.squeeze(preds_train_t[ix]))
plt.show()






































#Issue: If Y_train is dtype = 'bool', mask is dtype = 'float/int', we can't replace elements in Y_train with mask 
