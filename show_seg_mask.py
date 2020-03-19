import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
#patho = '/media/user1/externaldrive8TB/COR19_add/add_seg'
patho = '/home/user1/4TBHD/COR19/COR19_new/seg'
baseDir = os.path.normpath(patho)
files = glob(baseDir+'/*.npy')
for file in files:
    print(file)
    img = np.load(file)
    s = img.shape
    s = int(s[2]/2)
    plt.imshow(img[:,:,s])
    plt.show()