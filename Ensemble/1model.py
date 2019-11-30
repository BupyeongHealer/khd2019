import os
import argparse
import sys
import time
import random
import cv2
import numpy as np
import keras

from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from keras.layers import BatchNormalization, ReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import  ModelCheckpoint, ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model
import keras.backend.tensorflow_backend as K
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM
from keras.applications.xception import Xception


from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Activation

def cnn_sample(in_shape, num_classes=4):    # Example CNN

    ## 샘플 모델
    # 입력 데이터가 (390, 307, 3)의 크기일 때의 예시 코드

    model = Xception(input_shape=(h, w, 3), include_top=True, classes = 4, weights=None)
    
    out = model.layers[-2].output

    model2 = Sequential()
    model2.add()
    x = Dense(num_classes)(out)
    pred = Activation('softmax')(x)
    
    model = Model(inputs=model.input, outputs=pred)

    return model
    
