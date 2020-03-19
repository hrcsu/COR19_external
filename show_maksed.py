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
import evaluate

# model.summary()
from glob import glob
import matplotlib.pyplot as plt

#model = load_model('/home/user1/4TBHD/COR19/data/output/models/e4931b83-1390-4258-9ec9-1df23c5e9439-enet.h5')

basepath = '/home/user1/4TBHD/COR19/pre1'
basedir = os.path.normpath(basepath)
files = glob(basedir+'/COR189*.npy')
for file in files:
    image_file = file
    #fig = evaluate.plot_grad_cam(image_file, model, layer='top_conv', filter_idx=None, backprop_modifier="relu")
    #plt.show()
    print(file)
    img = np.load(file)
    print(img.max())
    plt.imshow(img,cmap=plt.cm.gray)
    plt.show()