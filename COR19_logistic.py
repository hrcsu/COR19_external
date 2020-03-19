import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import nrrd
from data import data, Dataset
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
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



#from keras.models import load_model
#model = load_model('/home/user1/4TBHD/COR19/data/output/models/a94dc1d4-2ae5-428d-bb64-a23bc966dc8a-enet.h5')
#model.summary()

#image_file = '/home/user1/4TBHD/COR19/pre1/COR013-145.npy'
#fig = evaluate.plot_grad_cam(image_file,model,layer='top_conv',filter_idx=None, backprop_modifier="relu")
#plt.show()

#train_features, train_label = predict(train_set)
#np.save('./COR_train_width.npy',train_features)
#np.save('./COR_train_label_width.npy',train_label)
#validation_features,validation_label = predict(validation_set)
#np.save('./COR_validation_width.npy',validation_features)
#np.save('./COR_validation_label_width.npy',validation_label)
train_features = np.load('./external_china_train_slice_prob_each.npy')
#print(train_features.shape)
train_label = np.load('./external_china_train_patient_label.npy')
#print(train_label.shape)
validation_features = np.load('./external_china_validation_slice_prob_each.npy')
validation_label = np.load('./external_china_validation_patient_label.npy')
#print(validation_features.shape)
#print(validation_label.shape)

test_features = np.load('./external_china_test_slice_prob_each.npy')
test_label = np.load('./external_china_test_patient_label.npy')
#test_features,test_label = predict(test_set)
#np.save('./COR_test_all.npy',test_features)
#np.save('./COR_test_label.npy',test_label)
#print(test_features)
import sklearn.metrics as metrics
import joblib
train = np.concatenate((train_features,validation_features),axis=0)
#train =validation_features
train_label = np.concatenate((train_label,validation_label),axis=0)
#train_label = validation_label

model = LogisticRegression(penalty = 'l1',solver = 'liblinear')
model.fit(train,train_label)
lg_pred = model.predict(train)
lg_pro = model.predict_proba(train)

print("Accuracy: " + repr(accuracy_score(train_label, lg_pred)))
print("Average Precision Score: " + repr(average_precision_score(train_label, lg_pred)))
print("Kappa: " + repr(cohen_kappa_score(train_label, lg_pred)))
print("Hamming Loss: " + repr(hamming_loss(train_label,lg_pred)))
prob_po = np.empty((len(lg_pro)))
for i in range(len(prob_po)):
    prob_po[i] = lg_pro[i,1]

print("AUC: " + repr(roc_auc_score(train_label, prob_po)))
print("Sensitivity: " + repr(recall_score(train_label, lg_pred)))
tn, fp, fn, tp = confusion_matrix(train_label, lg_pred).ravel()
print("Specificity: " + repr(tn / (tn + fp)))
fpr,tpr,th = precision_recall_curve(train_label,lg_pred)
print("PR auc"+repr(auc(tpr,fpr)))
print(tn,fp,fn,tp)

model = joblib.load('./logistic_model1.pkl')
lg_pred = model.predict(test_features)
#print(lg_pred)
''''''
lg_pro = model.predict_proba(test_features)


print("Accuracy: " + repr(accuracy_score(test_label,lg_pred )))
print("Average Precision Score: " + repr(average_precision_score(test_label, lg_pred)))
print("Kappa: " + repr(cohen_kappa_score(test_label, lg_pred)))
print("Hamming Loss: " + repr(hamming_loss(test_label,lg_pred)))
prob_po = np.empty((len(lg_pro)))
for i in range(len(prob_po)):
    prob_po[i] = lg_pro[i][1]
#print(prob_po)

np.save('test_label_patient_china.npy',test_label)
np.save('test_pro_patient_china.npy',prob_po)
print("AUC: " + repr(roc_auc_score(test_label, prob_po)))
print("Sensitivity: " + repr(recall_score(test_label, lg_pred)))
tn, fp, fn, tp = confusion_matrix(test_label, lg_pred).ravel()
print("Specificity: " + repr(tn / (tn + fp)))
fpr,tpr,th = precision_recall_curve(test_label,lg_pred)
print("PR auc"+repr(auc(tpr,fpr)))
print(tn,fp,fn,tp)

'''
Accuracy: 0.7074235807860262
Average Precision Score: 0.6086268594643749
Kappa: 0.42300026324696327
Hamming Loss: 0.2925764192139738
AUC: 0.772811059907834
Sensitivity: 0.819047619047619
Specificity: 0.6129032258064516
PR auc0.7719040480690744
76 48 19 86
'''

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,average_precision_score,cohen_kappa_score,hamming_loss,roc_auc_score,recall_score,confusion_matrix,precision_recall_curve,auc

#expert1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1]
#expert2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#expert3 = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
label_expert = [0]*124+[1]*105
#plot for tpot
fpr, tpr, thresholds= metrics.roc_curve(test_label, prob_po, pos_label=1)
auc_CT =metrics.auc(fpr, tpr)
plt.plot(fpr, tpr,
         label='Logistic Regression (area = {0:0.2f}, acc = 0.71)'
         ''.format(auc_CT),
         color='mediumpurple', linestyle=':', linewidth=3)



plt.plot([0,1],[0,1],color='y',linestyle=':',linewidth=2)

plt.legend(loc = 'lower right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('China.jpg')
plt.show()

fpr, tpr, thresholds= metrics.precision_recall_curve(test_label, prob_po, pos_label=1)
auc_CT =metrics.auc(tpr, fpr)
plt.plot(tpr, fpr,
         label='Logistic Regression (area = 0.77, acc = 0.71)'
         ''.format(auc_CT),
         color='mediumpurple', linestyle=':', linewidth=3)


plt.plot([0,1],[1,0],color='y',linestyle=':',linewidth=2)

plt.legend(loc = 'lower left')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('prroc_expert.jpg')
plt.show()
