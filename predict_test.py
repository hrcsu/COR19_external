import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import nrrd
from data import data, Dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
def predict(df):
    basepath =  '/home/user1/4TBHD/COR19_without_seg/preprocessed'
    features = []
    truth_label = []
    df.sort()
    print(df)
    label=np.zeros((1,))
    count = 0
    for name in df:
        print(name)
        label[0] = 0
        if 'COR' in name or 'SUB' in name:
            label[0] = 1
        vector = np.zeros(438)
        basedir = os.path.normpath(basepath)
        print(basedir)
        files = glob(basedir+'/'+name +'*.npy')
        files.sort()
        l = len(files)
        print(l)
        count = count+l
        print('hhhhh'+str(count))
        for i in range(l):
            img = np.load(files[i])
            s = img.shape
            if s[0] != 224:
                img = misc.imresize(img, (224, 224), 'bilinear')
            img = np.stack((img,img,img),axis =2)
            #img = img[np.newaxis,:,:,:]
            yield img, label[0]


from keras.models import load_model

model = load_model('/home/user1/4TBHD/COR19_exteranl_test/output/models/46a7de2d-c2af-4fa2-997f-4c9a4ed3193c-enet.h5')
#test_features,test_label = relist(predict(test_set))
test_features,test_label = relist(predict(test_set))
#test_features = np.load('./COR_test.npy')
#test_label = np.load('./COR_test_label.npy')

test_generator = Dataset(
    test_features,
    test_label,
    augment=False,
    shuffle=False,
    input_form='t1',
    seed=seed,
)

test_generator.reset()
test_results = evaluate.get_results(model, test_generator)
probabilities = list(evaluate.transform_binary_probabilities(test_results))
np.save('./test_slice_pro.npy',probabilities)
#test 5628
#validation 4593
#train 34585
lg_pred = np.zeros((len(probabilities)))
for i in range(len(probabilities)):
    if probabilities[i]<0.5:
        lg_pred[i] = 0
    else:
        lg_pred[i] = 1


print("Accuracy: " + repr(accuracy_score(test_label, lg_pred)))
print("Average Precision Score: " + repr(average_precision_score(test_label, lg_pred)))
print("Kappa: " + repr(cohen_kappa_score(test_label, lg_pred)))
print("Hamming Loss: " + repr(hamming_loss(test_label,lg_pred)))

print("AUC: " + repr(roc_auc_score(test_label, probabilities)))
print("Sensitivity: " + repr(recall_score(test_label, lg_pred)))
tn, fp, fn, tp = confusion_matrix(test_label, lg_pred).ravel()
print("Specificity: " + repr(tn / (tn + fp)))
fpr,tpr,th = precision_recall_curve(test_label,lg_pred)
print("PR auc"+repr(auc(tpr,fpr)))
print(tn,fp,fn,tp)

import matplotlib.pyplot as plt
plt.hist(probabilities, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
plt.show()

'''
test
Accuracy: 0.6584959377820985
Average Precision Score: 0.5392170752743425
Kappa: 0.3660021782802254
Hamming Loss: 0.3415040622179015
AUC: 0.8274291167847311
Sensitivity: 0.9498264975254566
Specificity: 0.4586325320012488
PR auc0.7582243440697195
11752 13872 882 16697
'''

'''
validation
Accuracy: 0.979765708200213
Average Precision Score: 0.9641553384880682
Kappa: 0.9592363476608321
Hamming Loss: 0.02023429179978701
AUC: 0.9985656128314447
Sensitivity: 0.9826860084230229
Specificity: 0.9773260359655981
PR auc0.9818449974387827
5000 116 74 4200
'''

'''
train
Accuracy: 0.972936244036432
Average Precision Score: 0.9879028058841112
Kappa: 0.9249982529396986
Hamming Loss: 0.027063755963568022
AUC: 0.9973922495379391
Sensitivity: 0.9704232725097218
Specificity: 0.9815074607830634
PR auc0.9938692390696116
7696 145 791 25953
'''



