"""
Author: Amol Kapoor

Description: Constants for text simplifier network.
"""

# Paths.
NORMAL_SENTENCE_PATH = 'data/normal_sentences.data'
SIMPLE_SENTENCE_PATH = 'data/simple_sentences.data'
NORMAL_VOCAB_PATH = 'data/normal_vocab.data'
SIMPLE_VOCAB_PATH = 'data/simple_vocab.data'
NORMAL_IDS_PATH = 'data/normal_ids.data'
SIMPLE_IDS_PATH = 'data/simple_ids.data'
NORMAL_IDS_TEST_PATH = 'data/normal_ids.data_test'
SIMPLE_IDS_TEST_PATH = 'data/simple_ids.data_test'
NORMAL_DOC_PATH = 'data/normal_doc.data'
CKPT_PATH = 'data/ckpt/'

# Data constants: IDs.
EMPT_TOKEN = 'EMPT'
EOS = 'EOS'
UNK_TOKEN = 'UNK_ID'
GO_TOKEN = 'GO'
TOKEN_LIST = [EMPT_TOKEN, EOS, UNK_TOKEN, GO_TOKEN]
EMPT_ID = 0
EOS_ID = 1
UNK_ID = 2
GO_ID = 3
EOS_GO_TOKEN_SIZE = 2

# Data constants: Vocab and Sets
MAX_LEN_IN = 35
MAX_LEN_OUT = 50
MAX_VOCAB_SIZE = 80000
TEST_SPLIT = 16 # Every 16th is put into test (about 10000)
TRAIN_VALID_SPLIT = 15 # Every fifteenth is put in validation (about 10000)

# Network constants.

#       Unchanged.
BATCH_SIZE =  64
NUM_SAMPLES = 512
LEARNING_RATE = 1
LEARNING_RATE_DECAY_FACTOR = 0.99
DECAY_POINT = 3 # If no improvement after this number of steps at ckpt, decay
IGNORE_STEPS = True
NUM_STEPS = 10 # If ignore steps is false, how long to train for
USE_SAMPLED_SOFTMAX = True # If not true this crashes comp
INPUT_KEEP_PROB = 0.5 # Dropout.
OUTPUT_KEEP_PROB = 0.5
MAX_GRAD_NORM = 5.0 # Use clipping

#       Hyperparams.
LAYER_SETUP = [256, 256, 256]
UNITS_PER_LAYER = LAYER_SETUP[len(LAYER_SETUP) - 1]
REG_CONST = 0.0000001

#       What to run. If all are false, defaults to input testing.
PIPE_TEST = False
TRAIN = False
TEST = False
DOC_TEST = True

# Debug constants.
DEBUG = False
STEPS_PER_CHECKPOINT = 100
CREATE_DATA = False
LOG_FILE_NAME = '256-256-256-7-graddesc'
TEST_BATCHES = 30
