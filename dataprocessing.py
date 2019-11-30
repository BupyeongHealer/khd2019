import os
import argparse
import sys
import time
import random
import keras
import cv2
import numpy as np

def image_preprocessing(im, rescale, resize_factor):
    ## 이미지 크기 조정 및 픽셀 범위 재설정
    h, w, c = 3072, 3900, 3
    nh, nw = int(h//resize_factor), int(w//resize_factor)
    # print(im.shape)

    res = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)

    if rescale == True:
        res = res / 255.

    return res


def Label2Class(label):     # one hot encoding (0-3 --> [., ., ., .])

    resvec = [0, 0, 0, 0]
    if label == 'AMD':		cls = 1;    resvec[cls] = 1
    elif label == 'RVO':	cls = 2;    resvec[cls] = 1
    elif label == 'DMR':	cls = 3;    resvec[cls] = 1
    else:					cls = 0;    resvec[cls] = 1		# Normal

    return resvec


def dataset_loader(img_path, idx, rescale, resize_factor):

    t1 = time.time()
    print('Loading training data...\n')
    if not ((resize_factor == 1.) and (rescale == False)):
        print('Image preprocessing...')
    if not resize_factor == 1.:
        print('Image size is 3072*3900*3')
        print('Resizing the image into {}*{}*{}...'.format(int(3072//resize_factor), int(3900//resize_factor), 3))
    if not rescale == False:
        print('Rescaling range of 0-255 to 0-1...\n')

    ## 이미지 읽기
    nor_p_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path + 'NOR/') for f in files if all(s in f for s in ['.jpg'])]
    amd_p_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path + 'AMD/') for f in files if all(s in f for s in ['.jpg'])]
    rvo_p_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path + 'RVO/') for f in files if all(s in f for s in ['.jpg'])]
    dmr_p_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path + 'DMR/') for f in files if all(s in f for s in ['.jpg'])]

    nor_p_list.sort()
    amd_p_list.sort()
    rvo_p_list.sort()
    dmr_p_list.sort()

    idx_step = int(len(nor_p_list)/4)
    if idx == 3:
        nor_p_list = nor_p_list[idx*idx_step:]
    else:  
        nor_p_list = nor_p_list[idx*idx_step:(idx+1)*idx_step]

    idx_step = int(len(amd_p_list)/4)
    if idx == 3:
        amd_p_list = amd_p_list[idx*idx_step:]
    else:  
        amd_p_list = amd_p_list[idx*idx_step:(idx+1)*idx_step]

    idx_step = int(len(rvo_p_list)/4)
    if idx == 3:
        rvo_p_list = rvo_p_list[idx*idx_step:]
    else:  
        rvo_p_list = rvo_p_list[idx*idx_step:(idx+1)*idx_step]

    idx_step = int(len(dmr_p_list)/4)
    if idx == 3:
        dmr_p_list = dmr_p_list[idx*idx_step:]
    else:  
        dmr_p_list = dmr_p_list[idx*idx_step:(idx+1)*idx_step]
    p_list = []
    for p in nor_p_list:
        p_list.append(p)
    for p in amd_p_list:
        p_list.append(p)
    for p in rvo_p_list:
        p_list.append(p)
    for p in dmr_p_list:
        p_list.append(p)
    
    num_data = len(p_list)

    images = []
    labels = []
    for i, p in enumerate(p_list):
        im = cv2.imread(p, 3)
        if not (resize_factor == 1.):
            im = image_preprocessing(im, rescale=rescale, resize_factor=resize_factor)
        images.append(im)

        # label 데이터 생성
        l = Label2Class(p.split('/')[-2])
        labels.append(l)

        print(i + 1, '/', num_data, ' image(s)')

    images = np.array(images)
    labels = np.array(labels)

    t2 = time.time()
    print('Dataset prepared for' ,t2 -t1 ,'sec')
    print('Images:' ,images.shape ,'np.array.shape(files, views, width, height)')
    print('Labels:', labels.shape, ' among 0-3 classes')

    return images, labels
