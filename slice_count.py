import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import nrrd
from data import data, Dataset
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from glob import glob
import evaluate

import scipy.ndimage

import efficientnet
import numpy as np
from sklearn.metrics import accuracy_score,average_precision_score,cohen_kappa_score,hamming_loss,roc_auc_score,recall_score,confusion_matrix,precision_recall_curve,auc
import math
import tensorflow as tf
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, concatenate
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2
from datetime import datetime
from config import config
from data import data, INPUT_FORM_PARAMETERS
import efficientnet.keras as efn
import matplotlib.pyplot as plt
#img.shape = (512,512)

import uuid
import traceback
import os
import numpy as np
import pandas
import nrrd
from glob import glob
import argparse
import random
from PIL import Image
import csv
from shutil import rmtree
from collections import defaultdict
from keras.preprocessing.image import ImageDataGenerator, Iterator
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


f = open('/home/user1/4TBHD/COR19_exteranl_test/train_set.txt', 'r')
fv = open('/home/user1/4TBHD/COR19_exteranl_test/validation_set.txt', 'r')
ft = open('/home/user1/4TBHD/COR19_exteranl_test/test_set.txt', 'r')
ft_test = ft.readlines()
fv_set = fv.readlines()
f_set = f.readlines()
test_set = []
validation_set = []
train_set = []
for line in ft_test:
    line = line.replace('\n', '')
    test_set.append(line)
    test_set.sort()
for line in fv_set:
    line = line.replace('\n', '')
    validation_set.append(line)
    validation_set.sort()
for line in f_set:
    line = line.replace('\n', '')
    train_set.append(line)
print(train_set)
print(test_set)
print(validation_set)
#img = np.load('/home/user1/4TBHD/COR19/COR19_new/masked_slices/COR001-176.npy')
#o,_ = nrrd.read('/home/user1/4TBHD/COR19/COR19_new/data/COR001.nrrd')
#mask = np.load('/home/user1/4TBHD/COR19/COR19_new/seg/COR001.npy')
#print(o.max())
#plt.imshow(mask[:,:,176]*(o[:,:,176]),cmap=plt.cm.gray)
#plt.show()
#max length = 438

def relist(l):
    l = list(l)
    if len(l) == 0:
        return l
    return [[k[i] for k in l] for i, _ in enumerate(l[0])]
seed = 'c07386a3-ce2e-4714-aa1b-3ba39836e82f'
import scipy.misc as misc
count_co = 0
count_no = 0
'''
41045
47258
'''
def predict(df):
    count_co = 0
    count_no = 0
    basepath =  '/home/user1/4TBHD/COR19_without_seg/preprocessed'
    basedir = os.path.normpath(basepath)
    features = []
    truth_label = []
    df.sort()
    print(df)
    label=np.zeros((1,))
    count = 0
    for name in df:
        #print(name)
        label[0] = 0
        vector = np.zeros(438)
        files = glob(basedir+'/'+name +'*.npy')
        files.sort()
        l = len(files)
        if 'COR' in name or 'SUB' in name:
            label[0] = 1
            count_co = count_co + l
        else:
            count_no = count_no+l
        #print(l)
        #count = count+l
        #print('hhhhh'+str(count))
        '''
        for i in range(l):
            img = np.load(files[i])
            s = img.shape
            if s[0] != 224:
                img = misc.imresize(img, (224, 224), 'bilinear')
            img = np.stack((img,img,img),axis =2)
            #img = img[np.newaxis,:,:,:]
            yield img, label[0]
        '''
    print(count_co)
    print(count_no)
predict(test_set)




