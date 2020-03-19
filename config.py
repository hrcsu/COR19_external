import os
import logging

class Config(object):
    IMAGE_SIZE = 224
    #MAX_VALUE =

    # Should change trials to 10
    TRIALS = 2
    BATCH_SIZE = 64

    EPOCHS = 50
    PATIENCE = 50
    SAMPLES_VALIDATION = 300
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.1

    DEVELOPMENT = True
    DEBUG = True
    PRINT_SQL = False
    SECRET = "example secret key"
    LOG_LEVEL = logging.DEBUG

    RAW_NRRD_ROOT = "/home/user1/4TBHD/COR19/data"
    INPUT_FORM = "t1"

    train = '/home/user1/4TBHD/COR19/data/slice_numpy/train'
    validation = '/home/user1/4TBHD/COR19/data/slice_numpy/validation'
    test ='/home/user1/4TBHD/COR19/data/slice_numpy/test'

    DATA = "/home/user1/4TBHD/COR19/"
    PREPROCESSED_DIR = os.path.join(DATA, "preprocessed")

    FEATURES_DIR = "./features"

    OUTPUT = "/home/user1/4TBHD/COR19_exteranl_test/output"
    DB_URL = "sqlite:///{}/results.db".format(OUTPUT)
    MODEL_DIR = os.path.join(OUTPUT, "models")
    STDOUT_DIR = os.path.join(OUTPUT, "stdout")
    STDERR_DIR = os.path.join(OUTPUT, "stderr")
    DATASET_RECORDS = os.path.join(OUTPUT, "datasets")

    MAIN_TEST_HOLDOUT = 0.2
    NUMBER_OF_FOLDS = 5
    SPLIT_TRAINING_INTO_VALIDATION = 0.1

config = Config()


