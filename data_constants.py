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
PROCESSED_ALIGNED_PATH = 'datamap.data'

# Data constants.
EMPT_TOKEN = '@@@ '
EOS = 'EOS'
UNK_TOKEN = 'UNK_ID'
MAX_LEN_IN = 35
MAX_LEN_OUT = 50
MAX_VOCAB_SIZE = 0

TOKEN_LIST = [EMPT_TOKEN, EOS, UNK_TOKEN]
EMPT_ID = 0
EOS_ID = 1
UNK_ID = 2
