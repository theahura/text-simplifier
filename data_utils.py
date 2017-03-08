"""API to handle data and associated labels."""

import data_constants as dc

import tensorflow as tf


def create_vocabulary(vocab_path, data_path, max_size):
    """
    Writes a vocab path to a file
    """
    vocab = {}
    f = open(data_path, 'r+')
    lines = f.readlines()
    for line in lines:
        line = tf.compat.as_bytes(line)
        tokens = line.strip().split(' ')
        for token in tokens:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    vocab_list = dc.TOKEN_LIST + sorted(vocab, key=vocab.get, reverse=True)
    if max_size and len(vocab_list) > max_size:
        vocab_list = vocab_list[:max_size]
    f.close()
    f = open(vocab_path, 'w+')
    for token in vocab_list:
        f.write(token + b'\n')
    f.close()

def get_vocabulary(vocab_path):
    """
    Returns a vocab map based on the input path
    """
    f = open(vocab_path, 'r+')
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in f.readlines()]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    f.close()
    return vocab, rev_vocab

def sentence_to_ids(sentence, vocabulary):
    """
    Returns a tokenized version of a sentence that the network can process.
    """
    words = sentence.strip().split(' ')
    return [vocabulary.get(w, dc.UNK_ID) for w in words]

def pad_data(token_sentence, is_normal):
    """
    Pads data to max length, with input data being front padded and reversed and
    output data being back padded and given go and eos tokens. Lines greater
    than max len are cut down.
    """
    if is_normal:
        token_sentence = token_sentence[:dc.MAX_LEN_IN]
        padding = [dc.EMPT_ID] * (dc.MAX_LEN_IN - len(token_sentence))
        return padding + token_sentence[::-1]
    else:
        token_sentence = token_sentence[:dc.MAX_LEN_OUT - 2]
        padding = [dc.EMPT_ID] * (dc.MAX_LEN_OUT - len(token_sentence))
        return [dc.GO_ID] + token_sentence + [dc.EOS_ID] + padding[:-1]


def data_to_ids(data_path, target_path, vocab_path, is_normal):
    """
    Stores a tokenized version of input data
    """
    f_data = open(data_path, 'r+')
    f_target = open(target_path, 'w+')
    lines = f_data.readlines()
    vocab, _ = get_vocabulary(vocab_path)

    for line in lines:
        line = tf.compat.as_bytes(line.strip())
        token_sentence = sentence_to_ids(line, vocab)
        padded_sentence = pad_data(token_sentence, is_normal)
        f_target.write(' '.join([str(tok) for tok in padded_sentence]) + '\n')
    f_data.close()
    f_target.close()

def split_to_test(data_path):
    f_data = open(data_path, 'r+')
    test_data = open(data_path + '_test', 'w+')
    lines = f_data.readlines()
    f_data.seek(0)
    for i, line in enumerate(lines):
        if i % dc.TEST_SPLIT == 0:
            test_data.write(line)
        else:
            f_data.write(line)
    f_data.truncate()
    f_data.close()
    test_data.close()

def process_data():
    """
    Preps data for text simplifier
    """
    create_vocabulary(dc.NORMAL_VOCAB_PATH, dc.NORMAL_SENTENCE_PATH,
                      dc.MAX_VOCAB_SIZE)
    create_vocabulary(dc.SIMPLE_VOCAB_PATH, dc.SIMPLE_SENTENCE_PATH,
                      dc.MAX_VOCAB_SIZE - 2)

    data_to_ids(dc.NORMAL_SENTENCE_PATH, dc.NORMAL_IDS_PATH,
                dc.NORMAL_VOCAB_PATH, True)
    data_to_ids(dc.SIMPLE_SENTENCE_PATH, dc.SIMPLE_IDS_PATH,
                dc.SIMPLE_VOCAB_PATH, False)

    split_to_test(dc.NORMAL_IDS_PATH)
    split_to_test(dc.SIMPLE_IDS_PATH)

if __name__ == '__main__':
    process_data()
