import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import nrrd
from data import data, Dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from glob import glob
#import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import math

from vis.utils import utils
import keras
from vis.visualization import visualize_cam, visualize_saliency, overlay
from vis.utils.utils import load_img, normalize, find_layer_idx
from keras.models import load_model, Model
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix, roc_auc_score
from sklearn import manifold
import pandas
from config import config
# model.summary()
from glob import glob
import matplotlib.pyplot as plt
import efficientnet.keras as efn
#46a7de2d-c2af-4fa2-997f-4c9a4ed3193c-enet

model = load_model('/home/user1/4TBHD/COR19/data/output/models/46a7de2d-c2af-4fa2-997f-4c9a4ed3193c-enet.h5')
model.summary()

def plot_grad_cam(image_file, model, layer, filter_idx=None, backprop_modifier="relu"):
    img = np.load(image_file)
    img = misc.imresize(img, (224, 224), 'bilinear')
    image = np.stack((img,img,img),axis=2)
    layer_idx = utils.find_layer_idx(model, 'res5c_branch2c')
    model.layers[layer_idx].activation = keras.activations.linear
    model = utils.apply_modifications(model)
    penultimate_layer_idx = utils.find_layer_idx(model, layer)
    #image = image[:,:,:]
    print(image.shape)

    class_idx = 1
    seed_input = image/255.
    grad_top1 = visualize_cam(model, layer_idx,filter_indices = None, seed_input = seed_input,
                              penultimate_layer_idx=None,  # None,
                              backprop_modifier=backprop_modifier,
                              grad_modifier=None)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(overlay(grad_top1,image),cmap="jet", alpha=0.8)
    ax[0].axis('off')
    ax[1].imshow(image)
    ax[1].axis('off')
    return fig

import scipy.misc as misc
basepath = '/home/user1/4TBHD/COR19_without_seg/preprocessed'
basedir = os.path.normpath(basepath)
files = glob(basedir+'/32787378-144.npy')
for file in files:
    image_file = file

    fig = plot_grad_cam(image_file, model, layer='res5c_branch2c', filter_idx=None, backprop_modifier=None)
    plt.show()

    print(file)
    #img = np.load(file)
    #plt.imshow(img,cmap=plt.cm.gray)
    #plt.show()
from vis.utils import utils
import keras
