class HyperParameter():
    EPOCH = 100
    BATCH_SIZE = 64
    RESIZE = (224,224)
    LOG_INTERVAL =50
    TRAIN_IMAGE_DIR = '/opt/ml/input/data/train/images'
    TRAIN_CSV_DIR = '/opt/ml/input/data/train/train.csv'
    TRAIN_FACE_IMAGE_DIR = '/opt/ml/face_input/train/images'
    TRAIN_FACE_CSV_DIR = '/opt/ml/face_input/train/train.csv'
    TEST_IMAGE_DIR = '/opt/ml/input/data/eval/images'
    TEST_CSV_DIR = '/opt/ml/input/data/eval/info.csv'
    SEED = 42
    DEFAULT_ROOT = '/opt/ml/pl_exp/'
    LEARNING_RATE = 1e-4
    VALIDATION_RATIO = 0.2
    NUM_CLASS=18
    NUM_MASK_CLASS=3
    NUM_GENDER_CLASS=2
    NUM_AGE_CLASS=3
