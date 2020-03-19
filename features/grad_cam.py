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
#ed72d40e-deb4-4b82-bbe7-947fa3c8facb-v2
#e4931b83-1390-4258-9ec9-1df23c5e9439-enet
model = load_model('/home/user1/4TBHD/COR19/data/output/models/e4931b83-1390-4258-9ec9-1df23c5e9439-enet.h5')
model.summary()
import evaluate
def plot_grad_cam(image_file, model, layer, filter_idx=None, backprop_modifier="relu"):
    img = np.load(image_file)
    image = np.stack((img,img,img),axis=2)
    layer_idx = utils.find_layer_idx(model, 'dense_6')
    model.layers[layer_idx].activation = keras.activations.linear
    model = utils.apply_modifications(model)
    penultimate_layer_idx = utils.find_layer_idx(model, layer)
    #image = image[:,:,:]
    print(image.shape)

    class_idx = 1
    seed_input = image/255.
    grad_top1 = visualize_cam(model, layer_idx, filter_indices = None,seed_input = seed_input,
                              penultimate_layer_idx=penultimate_layer_idx,  # None,2
                              backprop_modifier=backprop_modifier,
                              grad_modifier=None)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(overlay(grad_top1,image),cmap="jet", alpha=0.8)
    ax[0].axis('off')
    ax[1].imshow(image)
    ax[1].axis('off')
    return fig
seed = 'c07386a3-ce2e-4714-aa1b-3ba39836e82f'
#
basepath = '/home/user1/4TBHD/COR19/pre1'
basedir = os.path.normpath(basepath)
files = glob(basedir+'/*.npy')

layer_idx = utils.find_layer_idx(model, 'dense_6')
model.layers[layer_idx].activation = keras.activations.linear
model = utils.apply_modifications(model)
penultimate_layer_idx = utils.find_layer_idx(model, "top_conv")
for file in files:
    img = np.load(file)
    img = np.stack((img,img,img),axis =2)
    image = img

    # image = image[:,:,:]
    print(image.shape)

    class_idx = 1
    seed_input = image / 255.
    grad_top1 = visualize_cam(model, layer_idx, filter_indices=None, seed_input=seed_input,
                              penultimate_layer_idx=penultimate_layer_idx,  # None,
                              backprop_modifier="relu",
                              grad_modifier=None)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(overlay(grad_top1, image), cmap="jet", alpha=0.8)
    ax[0].axis('off')
    ax[1].imshow(image)
    ax[1].axis('off')
    plt.show()
    print(file)
    #img = np.load(file)conv1
    #plt.imshow(img,cmap=plt.cm.gray)
    #plt.show()
from vis.utils import utils
import keras
