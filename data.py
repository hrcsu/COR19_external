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

from config import config

clinical_features = [
    "age",
    "volume",
    "BMI",
    "missing_BMI",
    "Ca125",
    "missing_Ca125"
]

def all_input(t1, features, labels):
    t1_image = np.array(t1)
    t1_image = np.rollaxis(t1_image, 0, 3)
    return (t1_image, None), features, labels

def t1_input(t1, features, labels):
    t1_image = np.array(t1)
    t1_image = np.rollaxis(t1_image, 0, 3)
    return (t1_image, None), [], labels

def t1_features_input(t1,features, labels):
    t1_image = np.array(t1)
    t1_image = np.rollaxis(t1_image, 0, 3)
    return (t1_image, None), features, labels

def features_input(t1,features, labels):
    return (None, None), features, labels

INPUT_FORMS = {
    "all": all_input,
    "t1": t1_input,
    "t1-features": t1_features_input,
    "features": features_input,
}

INPUT_FORM_PARAMETERS = {
    "all": {
        "t1": True,
        "features": True,
    },
    "t1": {
        "t1": True,
        "features": False,
    },
    "t1-features": {
        "t1": True,
        "features": True,
    },
    "features": {
        "t1": False,
        "features": True,
    },
}

class Features(Iterator):
    def __init__(self, features, shuffle, seed):
        super(Features, self).__init__(len(features), config.BATCH_SIZE, shuffle, hash(seed) % 2**32 )
        self.features = np.array(features)

    def _get_batches_of_transformed_samples(self, index_array):
        return self.features[index_array]

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

class Dataset(object):
    def __init__(self, images, labels, augment=False, shuffle=False, seed=None, input_form="all"):
        self.shuffle = shuffle
        self.seed = seed
        self.augment = augment
        self.input_form = input_form

        self.parameters = INPUT_FORM_PARAMETERS[input_form]


        self.labels = labels

        self.features_size = 0
        self.n = len(labels)

        unique, index, inverse, counts = np.unique(self.labels, return_index=True, return_inverse=True, return_counts=True)

        self.y = inverse
        self.classes = inverse
        self.class_indices = { u: i for i, u in enumerate(unique) }

        separate_images = list(zip(*images))
        if self.parameters["t1"]:
            self.t1 = np.array(images)
            self.datagen = self._get_data_generator()

        self.reset()

    def __len__(self):
        return self.n

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):


        if self.parameters["t1"]:
            self.generator_t1 = self.datagen.flow(
                    x=self.t1,
                    y=self.y,
                    batch_size=config.BATCH_SIZE,
                    shuffle=self.shuffle,
                    seed=hash(self.seed) % 2**32,
                    )
        self.labels_generator = Features(self.y, self.shuffle, self.seed)

    def _get_data_generator(self):
        if self.augment:
            return ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
            )
        return ImageDataGenerator(
            rescale=1. / 255,
            )

    def next(self):
        labels = self.labels_generator.next()
        inputs = list()
        #if self.parameters["t2"]:
        #    inputs.append(self.generator_t2.next()[0])
        if self.parameters["t1"]:
            inputs.append(self.generator_t1.next()[0])
        if self.parameters["features"]:
            inputs.append(self.features_generator.next())
        if len(inputs) == 1:
            inputs = inputs[0]
        return (inputs, labels)

def outcome_feature(row):
    label = row["outcome"]
    features = [row[f] for f in clinical_features ]
    return label, features

LABEL_FORMS = {
    "outcome": outcome_feature,
}

def get_label_features(row, label="outcome"):
    """returns label, features, sample name"""
    return (*LABEL_FORMS[label](row), row.name)

def input_data_form(t1,features, labels, input_form=config.INPUT_FORM):
    images, features, labels = INPUT_FORMS[input_form](t1, features, labels)
    return images, features, labels



SHAPES_OUTPUT = """
SHAPES
    {}:"""
import random

#img.shape = (512,512)
#struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
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
import random

import scipy.misc as misc
def generate_from_features(df, input_form=config.INPUT_FORM, label_form="outcome", verbose=False, source=config.PREPROCESSED_DIR):
    files = []
    if df == 'train':
        files = train_set
        #random_index1 = random.sample(range(1, 34586), 7841)
    elif df == 'validation':
        files = validation_set
    else:
        files = test_set
    print(files)

    basedir = os.path.normpath('/home/user1/4TBHD/COR19_without_seg/preprocessed')
    files_list = glob(basedir + '/*.npy')
    files_list.sort()
    for file in files_list:
        name = file.split('/')
        name = name[-1].split('-')
        name = name[0]
        if name not in files:
            continue
        label = 0

        if 'COR' in name or 'SUB' in name:
            label = 0.75
        img = np.load(file)
        s = img.shape
        #print(s)
        if s[0] != 224:
            img =  misc.imresize(img,(224,224),'bilinear')
        #img = img[:,:,np.newaxis]
        img = np.stack((img,img,img),axis = 2)
        yield img, label


def sort(validation_fraction=0.2, test_fraction=0.1, seed=None, label_form="outcome"):
    f = pandas.read_pickle(config.FEATURES)
    train_fraction = 1 - validation_fraction - test_fraction

    remaining = f.copy()

    sort_dict = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }

    # calculate goal numbers for train/validation/test by label properties
    labels = f[label_form].unique()
    goal_sort = dict()
    for l in labels:
        label_fraction = len(remaining[remaining[label_form] == l])/len(remaining)
        for s in ["train", "validation", "test"]:
            goal_sort[(l, s)] = int(len(remaining) * label_fraction * sort_dict[s])

    all_train = list()
    all_validation = list()
    all_test = list()
    sorted_dict = {
        "train": all_train,
        "validation": all_validation,
        "test": all_test,
    }

    # get preassigned sorts
    train = f[f["sort"] == "train"]
    validation = f[f["sort"] == "validation"]
    test = f[f["sort"] == "test"]
    presort_dict = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    # recalculate goals based on preassigned sorts
    for s in ["train", "validation", "test"]:
        presorted = presort_dict[s]
        for l in labels:
            goal_sort[(l, s)] = max(0, goal_sort[(l, s)] - len(presorted[presorted[label_form] == l]))
    # add preassigned sorts and remove from lesions to sort
    all_train.append(train)
    all_validation.append(validation)
    all_test.append(test)
    remaining = remaining.drop(train.index)
    remaining = remaining.drop(validation.index)
    remaining = remaining.drop(test.index)

    # sort remaining lesions
    for l in labels:
        for s in ["train", "validation", "test"]:
            label_set = remaining[remaining[label_form] == l]
            label_set = label_set.sample(n = min(goal_sort[(l, s)], len(label_set)), random_state=(int(seed) % 2 ** 32))
            remaining = remaining.drop(label_set.index)
            sorted_dict[s].append(label_set)
    # append any left over
    all_train.append(remaining)

    train = pandas.concat(all_train)
    validation = pandas.concat(all_validation)
    test = pandas.concat(all_test)

    train.to_csv(os.path.join(config.DATASET_RECORDS, "{}-train.csv".format(str(seed))))
    validation.to_csv(os.path.join(config.DATASET_RECORDS, "{}-validation.csv".format(str(seed))))
    test.to_csv(os.path.join(config.DATASET_RECORDS, "{}-test.csv".format(str(seed))))

    return train, validation, test

def relist(l):
    l = list(l)
    if len(l) == 0:
        return l
    return [[k[i] for k in l] for i, _ in enumerate(l[0])]
import keras
def data(seed=None,
        input_form=config.INPUT_FORM,
        label_form="outcome",
        train_shuffle=True,
        validation_shuffle=False,
        test_shuffle=False,
        train_augment=True,
        validation_augment=False,
        test_augment=False,
        validation_split=config.VALIDATION_SPLIT,
        test_split=config.TEST_SPLIT,
        verbose=False,
        ):
    #train, validation, test = sort(validation_split, test_split, seed, label_form)

    test_images, test_labels = relist(generate_from_features('test', input_form=input_form, label_form=label_form, verbose=verbose))
    #test_labels = keras.utils.to_categorical(test_labels, dtype='float32')
    #test_labels = smooth_labels(test_labels,0.25)
    print(np.array(test_labels).shape)
    test_generator = Dataset(
            test_images,
            test_labels,
            augment=test_augment,
            shuffle=test_shuffle,
            input_form=input_form,
            seed=seed,
        )

    train_images, train_labels = relist(generate_from_features('train', input_form=input_form, label_form=label_form, verbose=verbose))
    print(np.array(train_images).shape)
    validation_images, validation_labels = relist(generate_from_features('validation', input_form=input_form, label_form=label_form, verbose=verbose))
    #validation_labels = keras.utils.to_categorical(validation_labels, num_classes=2, dtype='float32')
    #validation_labels = smooth_labels(validation_labels,0.25)
    #train_labels = keras.utils.to_categorical(train_labels, num_classes=2, dtype='float32')
    #train_labels = smooth_labels(train_labels,0.25)
    print(np.array(validation_images).shape)
    train_generator = Dataset(
            train_images,
            train_labels,
            augment=train_augment,
            shuffle=train_shuffle,
            input_form=input_form,
            seed=seed,
        )
    validation_generator = Dataset(
            validation_images,
            validation_labels,
            augment=validation_augment,
            shuffle=validation_shuffle,
            input_form=input_form,
            seed=seed,
        )
    return train_generator, validation_generator, test_generator

def load_from_features(
        features,
        input_form=config.INPUT_FORM,
        label_form="outcome",
        source=config.PREPROCESSED_DIR,
        shuffle=True,
        augment=True,
        verbose=False,
        ):
    images, features, labels, names = relist(generate_from_features(features, input_form=input_form, label_form=label_form, verbose=verbose, source=source))
    features = relist(features)

    generator = Dataset(
            images,
            features,
            labels,
            names,
            augment=augment,
            shuffle=shuffle,
            input_form=input_form,
            seed=0,
        )
    return generator


if __name__ == '__main__':
    data(uuid.uuid4())
