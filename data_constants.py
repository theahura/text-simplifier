"""
Author: Amol Kapoor

Description: Constants for text simplifier network.
"""

# Paths.
NORMAL_SENTENCE_PATH = 'data/normal_sentences.data'
SIMPLE_SENTENCE_PATH = 'data/simple_sentences.data'
NORMAL_IDS_PATH = 'data/normal_ids.data'
SIMPLE_IDS_PATH = 'data/simple_ids.data'
NORMAL_VOCAB_PATH = 'data/normal_vocab.data'
SIMPLE_VOCAB_PATH = 'data/simple_vocab.data'
CKPT_PATH = 'data/ckpt/'

# Data constants: IDs.
EMPT_TOKEN = '@@@ '
EOS = 'EOS'
UNK_TOKEN = 'UNK_ID'
GO_TOKEN = 'GO'
TOKEN_LIST = [EMPT_TOKEN, EOS, UNK_TOKEN, GO_TOKEN]
EMPT_ID = 0
EOS_ID = 1
UNK_ID = 2
GO_ID = 3
EOS_GO_TOKEN_SIZE = 2

# Data constants: Vocab
MAX_LEN_IN = 35
MAX_LEN_OUT = 50
MAX_VOCAB_SIZE = 80000

# Network constants.
UNITS_PER_LAYER = 256
LAYERS = 2
MAX_GRAD_NORM = 5.0
BATCH_SIZE =  64
LEARNING_RATE = 0.5
LEARNING_RATE_DECAY_FACTOR = 0.99
TRAIN_VALID_SPLIT = 4 # Every fourth is put in validation

# Debug constants.
STEPS_PER_CHECKPOINT = 200
CREATE_DATA = True
