class HyperParameter():
    EPOCH = 100
    BATCH_SIZE = 32
    RESIZE = (96,128)
    LOG_INTERVAL =20
    TRAIN_IMAGE_DIR = '/opt/ml/input/data/train/images'
    TRAIN_CSV_DIR = '/opt/ml/input/data/train/train.csv'
    TEST_IMAGE_DIR = '/opt/ml/input/data/eval/images'
    TEST_CSV_DIR = '/opt/ml/input/data/eval/info.csv'
    SAVE_DIR = '/opt/ml/model/run'
    SEED = 42
    LEARNING_RATE = 0.001
    VALIDATION_RATIO = 0.2
    NUM_CLASS=18
    NUM_MASK_CLASS=3
    NUM_GENDER_CLASS=2
    NUM_AGE_CLASS=3
