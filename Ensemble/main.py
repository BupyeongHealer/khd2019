import os
import argparse
import sys
import time
import random
import keras
import cv2
import numpy as np

from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization, ReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model

from keras.applications.densenet import DenseNet201
from keras.applications.xception import Xception

import keras.backend.tensorflow_backend as K
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM

from dataprocessing import image_preprocessing, image_preprocessing2, image_preprocessing3, image_preprocessing4, dataset_loader



## setting values of preprocessing parameters
RESIZE = 10.
RESCALE = True

def new_softmax(a) : 
    c = np.max(a) 
    exp_a = np.exp(a-c) 
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def bind_models(m1, m2):
# def bind_models(m1, m2, m3, m4):
    def save(dir_name):
        pass

    def load(dir_name):
        pass

    def infer(data, rescale=RESCALE, resize_factor=RESIZE):  ## test mode
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        X = []
        for i, d in enumerate(data):
            X.append(image_preprocessing(d, rescale, resize_factor))
        X = np.array(X)
        p1 = m1.predict(X)

        X2 = []
        for i, d in enumerate(data):
            X2.append(image_preprocessing2(d, rescale, resize_factor))
        X2 = np.array(X2)
        p2 = m2.predict(X2)

        # X3 = []
        # for i, d in enumerate(data):
        #     X3.append(image_preprocessing3(d, rescale, resize_factor))
        # X3 = np.array(X3)
        # p3 = m3.predict(X3)

        # X4 = []
        # for i, d in enumerate(data):
        #     X4.append(image_preprocessing4(d, rescale, resize_factor))
        # X4 = np.array(X4)
        # p4 = m4.predict(X4)

        pp1 = []
        for p in p1:
            p = new_softmax(p)
            pp1.append(p)

        pp2 = []
        for p in p2:
            p = new_softmax(p)
            pp2.append(p)

        # pp3 = []
        # for p in p3:
        #     p = new_softmax(p)
        #     pp3.append(p)

        # pp4 = []
        # for p in p4:
        #     p = new_softmax(p)
        #     pp4.append(p)

        pp1 = np.array(pp1)
        pp2 = np.array(pp2)
        # pp3 = np.array(pp3)
        # pp4 = np.array(pp4)

        # X = (pp1 + pp2 + pp3 + pp4) / 4
        X = (pp1 + pp2) / 2

        pred = np.argmax(X, axis=-1)     

        print('Prediction done!\n Saving the result...')
        return pred

    nsml.bind(save=save, load=load, infer=infer)

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        # model.save_weights(file_path,'model')
        print('model saved!')

    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded!')

    def infer(data, rescale=RESCALE, resize_factor=RESIZE):  ## test mode
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        X = []
        for i, d in enumerate(data):
            X.append(image_preprocessing(d, rescale, resize_factor))
        X = np.array(X)

        pred = model.predict(X)    
        print('Prediction done!\n Saving the result...')
        return pred

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    print("start!")
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=1)                          # epoch 
    args.add_argument('--batch_size', type=int, default=8)                      # batch size 
    args.add_argument('--num_classes', type=int, default=4)                     # DO NOT CHANGE num_classes, class �섎뒗 ��긽 4

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit�쇰븣 �대떦媛믪씠 test濡� �ㅼ젙�⑸땲��.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 紐낅졊�대� �낅젰�좊븣�� 泥댄겕�ъ씤�몃줈 �ㅼ젙�⑸땲��. 泥댄겕�ъ씤�� �듭뀡�� �덉＜硫� 留덉�留� wall time �� model �� 媛��몄샃�덈떎.')
    args.add_argument('--pause', type=int, default=0, help='model �� load �좊븣 1濡� �ㅼ젙�⑸땲��.')

    config = args.parse_args()
    print("args config!")

    seed = 1234
    np.random.seed(seed)

    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes

    """ Model """
    
    learning_rate = 1e-4

    h, w = int(3072//RESIZE), int(3900//RESIZE)

    # model = cnn_sample(in_shape=(h, w, 3), num_classes=num_classes)
    # adam = optimizers.Adam(lr=learning_rate, decay=1e-5)                    # optional optimization
    # sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])

    model_1 = DenseNet201(input_shape=(h, w, 3), include_top=True, classes = 4, weights=None)
    print("model1 bind start")
    bind_model(model=model_1)
    nsml.load('38', load_fn=None, session='team135/KHD2019_FUNDUS/124')
    print("model1 load end")

    model_2 = Xception(input_shape=(h, w, 3), include_top=True, classes = 4, weights=None)
    print("model2 bind start")
    bind_model(model=model_2)
    nsml.load('27', load_fn=None, session='team135/KHD2019_FUNDUS/129')
    print("model2 load end")

    # model_3 = Xception(input_shape=(h, w, 4), include_top=True, classes = 4, weights=None)
    # print("model3 bind start")
    # bind_model(model=model_3)
    # nsml.load('7', load_fn=None, session='team135/KHD2019_FUNDUS/174')
    # print("model3 load end")

    # model_4 = Xception(input_shape=(h, w, 3), include_top=True, classes = 4, weights=None)
    # print("model4 bind start")
    # bind_model(model=model_4)
    # nsml.load('26', load_fn=None, session='team135/KHD2019_FUNDUS/175')
    # print("model4 load end")

    bind_models(model_1, model_2)
    # bind_models(model_1, model_2, model_3, model_4)

    print('Inferring Start...')
    nsml.paused(scope=locals())