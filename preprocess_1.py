import os
import nrrd
import numpy as np
from glob import glob

import scipy
import scipy.ndimage

def normalize(image):
    MIN_BOUND = -400
    MAX_BOUND = -240
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

'''
def get_pixels_hu(CT_images):
    CT_images = CT_images.astype(np.int16)
    CT_images[CT_images==-2000] = 0
    return CT_images
'''

def resample_img(image, target_shape, mode='nearest'):
    """Resample the image to target_shape
    """
    print(target_shape)
    resize_factor = np.array(target_shape)/image.shape
    resampled = scipy.ndimage.interpolation.zoom(image, resize_factor,
                                                 mode=mode)
    return resampled
import matplotlib.pyplot as plt


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image

def window(x, wl, ww, convert_8bit=True):
    x = np.clip(x, wl-ww/2, wl+ww/2)
    if convert_8bit:
      x = x - np.min(x)
      x = x / np.max(x)
      x = (x * 255).astype('uint8')
    return x


def crop_image(image, display=False):
    # Create a mask with the background pixels
    mask = image == 0

    # Find the brain area
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    # Remove the background
    croped_image = image[top_left[0]:bottom_right[0],
                   top_left[1]:bottom_right[1]]

    return croped_image


def add_pad(image, new_height=512, new_width=512):
    height, width = image.shape

    final_image = np.zeros((new_height, new_width))

    pad_left = int((new_width - width) / 2)
    pad_top = int((new_height - height) / 2)

    # Replace the pixels with the image's pixels
    final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image

    return final_image


def win_scale(data, wl, ww, dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """

    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1] - 1)

    data_new[data <= (wl - ww / 2.0)] = out_range[0]
    data_new[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] = \
        ((data[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] - (wl - 0.5)) / (ww - 1.0) + 0.5) * (
                    out_range[1] - out_range[0]) + out_range[0]
    data_new[data > (wl + ww / 2.0)] = out_range[1] - 1

    return data_new.astype(dtype)

def run():
    patho = '/media/user1/Elements/2018 second half'
    basedir = os.path.normpath(patho)
    files = glob(basedir + '/*.nrrd')
    print("Preprocessing!")
    for file in files:

        print(file)
        name = file.split('/')
        name = name[-1]
        to_path = '/home/user1/4TBHD/COR19_without_seg/' + name
        if os.path.exists(to_path):
           continue
        t1_nrrd, _ = nrrd.read(file)
        #t1_nrrd = get_pixels_hu(t1_nrrd)
        s = t1_nrrd.shape
        if s[0]!=512:
            print(file)
        if 'COR' in name:
            t1_nrrd = t1_nrrd[:,:,::-1]
        #t1_nrrd = resample_img(t1_nrrd,(512,512,s[2]))
        #plt.imshow(t1_nrrd[:, :, 50], cmap=plt.cm.gray)
        #plt.show()
        #t1_nrrd = window_image(t1_nrrd,-400,200)
        #mask = np.load('/home/user1/4TBHD/COR19/COR19_new/seg/'+name[0:-5]+'.npy')
        t1_nrrd = window(t1_nrrd,-400,1500,convert_8bit=True)
        #plt.imshow(t1_nrrd[:, :, 50], cmap=plt.cm.gray)
        #plt.show()
        #t1_nrrd = normalize(t1_nrrd)
        #t1_nrrd = zero_center(t1_nrrd)
        #print(t1_nrrd[256,256,])

        nrrd.write(to_path, t1_nrrd)

run()
