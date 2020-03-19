import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import nrrd
from data import data, Dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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


seed = 'c07386a3-ce2e-4714-aa1b-3ba39836e82f'
import scipy.misc as misc
def predict(df):
    basepath = '/home/user1/4TBHD/COR19_without_seg/preprocessed'
    features = []
    truth_label = []
    df.sort()
    print(df)
    label=np.zeros((1,))
    for name in df:
        print(name)
        label[0] = 0
        if 'COR' in name or 'SUB' in name:
            label[0] = 1
        vector = np.zeros(800)
        basedir = os.path.normpath(basepath)
        print(basedir)
        files = glob(basedir+'/'+name +'*.npy')
        files.sort()
        l = len(files)
        print(l)
        if l==0:
            break
        for i in range(l):
            img = np.load(files[i])
            s = img.shape
            if s[0] != 224:
                img = misc.imresize(img, (224, 224), 'bilinear')
            img = np.stack((img,img,img),axis =2)
            img = img[np.newaxis,:,:,:]
            #print(img.shape)
            test_generator = Dataset(
                img,
                label,
                augment=False,
                shuffle=False,
                input_form='t1',
                seed=seed,
            )
            test_generator.reset()
            test_results = evaluate.get_results(model, test_generator)
            probabilities = list(evaluate.transform_binary_probabilities(test_results))
            vector[i] = probabilities[0]
            print(probabilities[0])
        print(len(vector))
        features.append(vector)
        truth_label.append(label[0])
    return features,truth_label

from keras.models import load_model
model = load_model('/home/user1/4TBHD/COR19_exteranl_test/output/models/46a7de2d-c2af-4fa2-997f-4c9a4ed3193c-enet.h5')

train_features, train_label = predict(train_set)
np.save('./external_china_train_slice_prob_each.npy',train_features)
np.save('./external_china_train_patient_label.npy',train_label)
validation_features,validation_label = predict(validation_set)
np.save('./external_china_validation_slice_prob_each.npy',validation_features)
np.save('./external_china_validation_patient_label.npy',validation_label)
'''


'''
test_features,test_label = predict(test_set)
np.save('./external_china_test_slice_prob_each.npy',test_features)
np.save('./external_china_test_patient_label.npy',test_label)

import sklearn.metrics as metrics




'''
Accuracy: 0.9893333333333333
Average Precision Score: 0.9869753639417692
Kappa: 0.9786659081211776
Hamming Loss: 0.010666666666666666
AUC: 0.9955618776671408
Sensitivity: 0.9842105263157894
Specificity: 0.9945945945945946
PR auc0.9934456886898096
184 1 3 187
Accuracy: 0.9024390243902439
Average Precision Score: 0.8721466541588493
Kappa: 0.7955112219451372
Hamming Loss: 0.0975609756097561
AUC: 0.9852941176470588
Sensitivity: 0.9583333333333334
Specificity: 0.8235294117647058
PR auc0.9336694809255784
14 3 1 23

'''
'''
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,average_precision_score,cohen_kappa_score,hamming_loss,roc_auc_score,recall_score,confusion_matrix,precision_recall_curve,auc

expert1 = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
expert2 = [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
expert3 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
expert4 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
expert5 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
expert6 = [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
expert7 = [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label_expert = [1]*24+[0]*17

#plot for tpot
fpr, tpr, thresholds= metrics.roc_curve(test_label, prob_po, pos_label=1)
auc_CT =metrics.auc(fpr, tpr)
plt.plot(fpr, tpr,
         label='Logistic Regression (area = {0:0.2f}, acc = 0.90)'
         ''.format(auc_CT),
         color='mediumpurple', linestyle=':', linewidth=3)

#plot for expert1
fpr, tpr, thresholds= metrics.roc_curve(label_expert, expert1, pos_label=1)
acc_1 = accuracy_score(label_expert, expert1)
plt.scatter(fpr[1],tpr[1], marker = 's', color = 'black', label='Expert1(acc = {0:0.2f})'
            ''.format(acc_1), s = 30)

#plot for expert2
fpr, tpr, thresholds= metrics.roc_curve(label_expert, expert2, pos_label=1)
acc_2 = accuracy_score(label_expert, expert2)
plt.scatter(fpr[1], tpr[1], marker = 'o', color = 'black', label='Expert2(acc = {0:0.2f})'
            ''.format(acc_2), s = 30)

#plot for expert3
fpr, tpr, thresholds= metrics.roc_curve(label_expert, expert3, pos_label=1)
acc_3 = accuracy_score(label_expert, expert3)
plt.scatter(fpr[1], tpr[1], marker = 'v', color = 'black', label='Expert3(acc = {0:0.2f})'
            ''.format(acc_3), s = 30)

#plot for expert4
fpr, tpr, thresholds= metrics.roc_curve(label_expert, expert4, pos_label=1)
acc_4 = accuracy_score(label_expert, expert4)
plt.scatter(fpr[1], tpr[1], marker = '<', color = 'black', label='Expert4(acc = {0:0.2f})'
            ''.format(acc_4), s = 30)

fpr, tpr, thresholds= metrics.roc_curve(label_expert, expert5, pos_label=1)
acc_4 = accuracy_score(label_expert, expert5)
plt.scatter(fpr[1], tpr[1], marker = '>', color = 'black', label='Expert5(acc = {0:0.2f})'
            ''.format(acc_4), s = 30)

fpr, tpr, thresholds= metrics.roc_curve(label_expert, expert6, pos_label=1)
acc_4 = accuracy_score(label_expert, expert6)
plt.scatter(fpr[1], tpr[1], marker = '2', color = 'black', label='Expert6(acc = {0:0.2f})'
            ''.format(acc_4), s = 30)

fpr, tpr, thresholds= metrics.roc_curve(label_expert, expert7, pos_label=1)
acc_4 = accuracy_score(label_expert, expert7)
plt.scatter(fpr[1], tpr[1], marker = '*', color = 'black', label='Expert7(acc = {0:0.2f})'
            ''.format(acc_4), s = 30)

plt.plot([0,1],[0,1],color='y',linestyle=':',linewidth=2)

plt.legend(loc = 'lower right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_expert.jpg')
plt.show()

fpr, tpr, thresholds= metrics.precision_recall_curve(test_label, prob_po, pos_label=1)
auc_CT =metrics.auc(tpr, fpr)
plt.plot(tpr, fpr,
         label='Logistic Regression (area = {0:0.2f}, acc = 0.90)'
         ''.format(auc_CT),
         color='mediumpurple', linestyle=':', linewidth=3)

#plot for hand optimization

#plot for expert1
fpr, tpr, thresholds= metrics.precision_recall_curve(label_expert, expert1, pos_label=1)
acc_1 = accuracy_score(label_expert, expert1)
plt.scatter(tpr[1],fpr[1], marker = 's', color = 'black', label='Expert1(acc = {0:0.2f})'
            ''.format(acc_1), s = 30)

#plot for expert2
fpr, tpr, thresholds= metrics.precision_recall_curve(label_expert, expert2, pos_label=1)
acc_2 = accuracy_score(label_expert, expert2)
plt.scatter(tpr[1], fpr[1], marker = 'o', color = 'black', label='Expert2(acc = {0:0.2f})'
            ''.format(acc_2), s = 30)

#plot for expert2
fpr, tpr, thresholds= metrics.precision_recall_curve(label_expert, expert3, pos_label=1)
acc_3 = accuracy_score(label_expert, expert3)
plt.scatter(tpr[1], fpr[1], marker = 'v', color = 'black', label='Expert3(acc = {0:0.2f})'
            ''.format(acc_3), s = 30)

#plot for expert2
fpr, tpr, thresholds= metrics.precision_recall_curve(label_expert, expert4, pos_label=1)
acc_4 = accuracy_score(label_expert, expert4)
plt.scatter(tpr[1], fpr[1], marker = '<', color = 'black', label='Expert4(acc = {0:0.2f})'
            ''.format(acc_4), s = 30)

fpr, tpr, thresholds= metrics.precision_recall_curve(label_expert, expert5, pos_label=1)
acc_4 = accuracy_score(label_expert, expert5)
plt.scatter(tpr[1], fpr[1], marker = '>', color = 'black', label='Expert5(acc = {0:0.2f})'
            ''.format(acc_4), s = 30)

fpr, tpr, thresholds= metrics.precision_recall_curve(label_expert, expert6, pos_label=1)
acc_4 = accuracy_score(label_expert, expert6)
plt.scatter(tpr[1], fpr[1], marker = '2', color = 'black', label='Expert6(acc = {0:0.2f})'
            ''.format(acc_4), s = 30)

fpr, tpr, thresholds= metrics.precision_recall_curve(label_expert, expert7, pos_label=1)
acc_4 = accuracy_score(label_expert, expert7)
plt.scatter(tpr[1], fpr[1], marker = '*', color = 'black', label='Expert7(acc = {0:0.2f})'
            ''.format(acc_4), s = 30)

plt.plot([0,1],[1,0],color='y',linestyle=':',linewidth=2)

plt.legend(loc = 'lower left')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('prroc_expert.jpg')
plt.show()
'''