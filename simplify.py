""" 
Where training and simplification actually happen. Influenced by tf seq2seq
tutorial.
"""

import nltk
import numpy as np
import math
import os
import sys
import tensorflow as tf
import time

import data_constants as dc
import data_utils
import model as sm

def read_data(normal_path, simple_path, is_test=False):
    """Reads data from files and returns as list of tuples 

    Args:
        normal_path: str, the path to ids for normal wikipedia data
        simple_path: str, same as normal path but simplewiki
        is_test: whether to return train/valid split, or just test data

    Return: 
        [(normal data : simple data)]
    """
    normal_f = open(normal_path, 'r+')
    simple_f = open(simple_path, 'r+')

    normal_line, simple_line  = normal_f.readline(), simple_f.readline()
    train_set = []
    valid_set = []
    count = 0

    while normal_line and simple_line:
        normal_ids = [int(x) for x in normal_line.split()]
        simple_ids = [int(x) for x in simple_line.split()]
        
        if count % dc.TRAIN_VALID_SPLIT == 0 and not is_test:
            valid_set.append([normal_ids, simple_ids])
        else:
            train_set.append([normal_ids, simple_ids])
        normal_line, simple_line = normal_f.readline(), simple_f.readline()
        count += 1

    normal_f.close()
    simple_f.close()

    return train_set, valid_set

def create_model(session, feed_previous):
    """Create translation model. Loads checkpoints if available.
    
    Args:
        session: tf sess to use for the model
        feed_previous: whether or not the model should construct a graph where
            previous inputs are fed in by hand instead of predicted

    Return:
        tf simplifier model
    """
    
    normal_vocab_size, simple_vocab_size = 0, 0

    with open(dc.NORMAL_VOCAB_PATH, 'r+') as f:
        normal_vocab_size = len(f.readlines())
    with open(dc.SIMPLE_VOCAB_PATH, 'r+') as f:
        simple_vocab_size = len(f.readlines()) + 2

    model = sm.SimplifierModel(
                normal_vocab_size,
                simple_vocab_size, 
                dc.UNITS_PER_LAYER,
                dc.MAX_GRAD_NORM,
                dc.BATCH_SIZE,
                dc.LEARNING_RATE,
                dc.LEARNING_RATE_DECAY_FACTOR,
                dc.NUM_SAMPLES,
                feed_previous=feed_previous,
                dtype=tf.float32)

    ckpt = tf.train.get_checkpoint_state(dc.CKPT_PATH)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print "loading old model"
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print 'loading new model'
        session.run(tf.global_variables_initializer())
    return model

def train(create_data, log_file):
    """Trains a english to simple english translation model.
    
    Args:
        create_data: whether to load data from databases or not on startup.
        log_file: where to store training data outputs.
    """

    if os.path.isfile('./' + log_file):
        raise ValueError('log file already exists')

    if create_data:
        data_utils.process_data()

    with tf.Session() as sess, open(log_file, 'w+') as log:
        print 'Opening log file'
        fields = ['step', 'step-time', 'batch-loss', 'batch-perplexity',
                'learnrate', 'val-loss']
        log.write(','.join(fields) + '\n')

        print 'creating model'
        model = create_model(sess, False)

        print 'reading data'
        train, valid = read_data(dc.NORMAL_IDS_PATH, dc.SIMPLE_IDS_PATH)

        print 'entering training loop'
        step_time, loss = 0.0, 0.0
        current_step = 0
        prev_losses = []

        # Training loop
        while current_step < dc.NUM_STEPS or dc.IGNORE_STEPS:
            start_time = time.time()
            encoder_in, decoder_in, target_weights = model.get_batch(train)
            step_loss, _ = model.step(sess, encoder_in, decoder_in,
                                         target_weights, False)
            step_time += (time.time() - start_time)/dc.STEPS_PER_CHECKPOINT

            loss += step_loss / dc.STEPS_PER_CHECKPOINT
            current_step += 1
            
            if current_step < dc.STEPS_PER_CHECKPOINT:
                print "Step: %f" % current_step
                print "Loss: %f" % step_loss
                print "Learning: %f" % model.learning_rate.eval()

            # Every some amount of steps, output stats and check validation loss
            if current_step % dc.STEPS_PER_CHECKPOINT == 0:
                learnrate = model.learning_rate.eval()
                perplex = math.exp(float(loss)) if loss < 300 else float("inf")
                step = model.global_step.eval()
                print "step %d loss %f plex: %f learnrate %f step-time: %f" % (
                        step, loss, perplex, learnrate, step_time)

                if len(prev_losses) > 2 and loss > max(
                        prev_losses[-1 * dc.DECAY_POINT:]):
                    sess.run(model.learning_rate_decay_op)
                prev_losses.append(loss)
                checkpoint_path = os.path.join(dc.CKPT_PATH, 'simplify.ckpt')
                model.saver.save(sess, checkpoint_path,
                        global_step=model.global_step)

                encoder_in, decoder_in, target_weights = model.get_batch(valid)
                val_loss, outputs = model.step(sess, encoder_in, decoder_in,
                                                  target_weights, True)

                if dc.DEBUG:
                    print "ENCODER LEN"
                    print len(encoder_in[0])
                    print "OUTPUT LENs"
                    print len(outputs)
                    print len(outputs[0])
                    print len(outputs[0][0])
                outputs = [int(np.argmax(logit, axis=1)[0])
                        for logit in outputs]
                
                if dc.DEBUG:
                    print outputs

                print "validation loss: %f" % val_loss

                fields = [step, step_time, loss, perplex, learnrate, val_loss]
                log.write(','.join(map(str, fields)) + '\n')

                step_time, loss = 0.0, 0.0
            sys.stdout.flush()

def test(log_file):
    """Get BLEU metrics for test data.
    
    Args:
        log_file: where to store BLEU data.
    """

    if os.path.isfile('./' + log_file + '_test'):
        raise ValueError('log file already exists')

    with tf.Session() as sess, open(log_file + '_test', 'w+') as log:
        log.write('batch,bleu\n')
        model = create_model(sess, True)

        print 'reading data'
        test, _ = read_data(dc.NORMAL_IDS_TEST_PATH, dc.SIMPLE_IDS_TEST_PATH,
                            True)

        avg_bleu = 0
        for batch in xrange(dc.TEST_BATCHES):
            encoder_in, decoder_in, target_weights = model.get_batch(test)
            _, output_logits = model.step(sess, encoder_in, decoder_in,
                                             target_weights, True)

            if dc.DEBUG:
                print output_logits[0].shape

            outputs = [np.argmax(logit, axis=1) for logit in output_logits]

            if dc.DEBUG:
                print len(outputs)
                print outputs

            def get_bleu(h, r):
                return nltk.translate.bleu_score.sentence_bleu([r], h)

            avg_bleu_per_batch = 0

            if dc.DEBUG:
                print 'hypothesis'
                print [pos[0] for pos in outputs]
                print [pos[0] for pos in encoder_in]

            for i in xrange(dc.BATCH_SIZE):
                h = [pos[i] for pos in outputs]
                r = [pos[i] for pos in encoder_in]
                avg_bleu_per_batch += get_bleu(h, r)/dc.BATCH_SIZE
            
            print "Batch %d bleu %f" % (batch, avg_bleu_per_batch)
            log.write("%d,%f\n" % (batch, avg_bleu_per_batch)) 
            avg_bleu += avg_bleu_per_batch

        avg_bleu = avg_bleu/dc.TEST_BATCHES
        print "Avg bleu: %f" % avg_bleu

def pipe_sentence(sentence, normal_vocab, rev_simple_vocab, sess, model):
    """Helper method to pipe a single sentence into the model.

    Args:
        sentence: str, unformatted sentence
        normal_vocab: {word : id}, vocab for inputs
        rev_simple_vocab: [] with words matching index for id, vocab for outputs
        sess: tf session
        model: tf model for simplification

    Return:
       str, predicted simplified sentence 
    """
    token_sentence = data_utils.sentence_to_ids(sentence, normal_vocab)

    if dc.DEBUG:
        print token_sentence

    padded_sentence = data_utils.pad_data(token_sentence, True)

    if dc.DEBUG:
        print padded_sentence

    # Decoder input doesnt matter because the predicted value is fed in.
    decoder = [dc.GO_ID] + [dc.EMPT_ID]*(dc.MAX_LEN_OUT - 1)

    input_token_sentence = [(padded_sentence, decoder)]
    encoder_in, decoder_in, target_weights = model.get_batch(
            input_token_sentence)

    if dc.DEBUG:
        print encoder_in
        print decoder_in
        print target_weights

    loss, output_logits = model.step(sess, encoder_in, decoder_in,
            target_weights, True)

    if dc.DEBUG:
        print "Loss: %f" % loss
        print output_logits
        print len(output_logits)
        print len(output_logits[0])
        print len(output_logits[0][0])
        print output_logits[0]
        print np.max(output_logits[0], axis=1)

    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    if dc.EOS_ID in outputs:
        outputs = outputs[:outputs.index(dc.EOS_ID)]

    if dc.DEBUG:
        print len(outputs)
        print outputs
    
    translation = " ".join([tf.compat.as_str(rev_simple_vocab[output])
        for output in outputs])

    if dc.DEBUG:
        print(translation)
    return translation

def doc_test(log_file):
    """Test the model qualitatively on a document.
    
    Args:
        log_file: where to store the output document.
    """
    if os.path.isfile('./' + log_file + '_doc'):
        raise ValueError('log file already exists')

    # The temp file created here splits the input document into line by line
    # representations to make it easier to see how translations were done on a
    # sentence level.
    with tf.Session() as sess, open(log_file + '_doc', 'w+') as log, open(
            dc.NORMAL_DOC_PATH, 'r+') as doc_file, open(
                    'temp_doc', 'w+') as temp:
        doc_to_translate = doc_file.readline()
        sentences = [sentence + '.' for sentence in doc_to_translate.split('.')]
        model = create_model(sess, True)
        model.batch_size = 1
        normal_vocab, _ = data_utils.get_vocabulary(dc.NORMAL_VOCAB_PATH)
        _, rev_simple_vocab = data_utils.get_vocabulary(dc.SIMPLE_VOCAB_PATH)
        for sentence in sentences:
            translation = pipe_sentence(sentence, normal_vocab,
                    rev_simple_vocab, sess, model)
            log.write(translation + '\n')
            temp.write(sentence + '\n')

def input_test():
    """Input your own sentence and see how the model interprets it."""
    with tf.Session() as sess:
        model = create_model(sess, True)
        model.batch_size = 1
        
        normal_vocab, _ = data_utils.get_vocabulary(dc.NORMAL_VOCAB_PATH)
        _, rev_simple_vocab = data_utils.get_vocabulary(dc.SIMPLE_VOCAB_PATH)

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            translation = pipe_sentence(sentence, normal_vocab,
                    rev_simple_vocab, sess, model)

            print translation

            print "> "
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def pipe_test():
  """Test the model pipeline. Used for debugging."""

  with tf.Session() as sess:
    print("Pipeline test for neural translation model")
    # Create model with vocabularies of 10, 2 layers of 32, batch size 32.
    dc.MAX_LEN_IN = 2
    dc.MAX_LEN_OUT = 2
    model = sm.SimplifierModel(10, 10, 32, 2, 5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.global_variables_initializer())

    data_set = ([([1, 7], [2, 2]), ([8, 3], [4, 4]), ([9, 5], [6, 6])])
    for _ in xrange(500):  # Train the fake model for 50 steps.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set)
      step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
              target_weights, False)
      print "Loss %f" % step_loss
      
    model.batch_size = 1
    input_token = [([1, 7], [0, 0])]
    encode, decode, weights = model.get_batch(input_token)
    _, output_logits = model.step(sess, encode, decode, weights, True)
    print output_logits
    print len(output_logits)
    print len(output_logits[0])
    print len(output_logits[0][0])
    print np.argmax(output_logits[0], axis=1)
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    print len(outputs)
    print outputs

    input_token = [([9, 5], [0, 0])]
    encode, decoder, weights = model.get_batch(input_token)
    _, output_logits = model.step(sess, encode, decode, weights, True)
    outputs = [int(np.argmax(logit, axis=1)[0]) for logit in output_logits]
    print len(outputs)
    print outputs



if __name__ == "__main__":
    if dc.PIPE_TEST:
        pipe_test()
    elif dc.TRAIN:
            train(dc.CREATE_DATA, dc.LOG_FILE_NAME)
    elif dc.TEST:
        test(dc.LOG_FILE_NAME)
    elif dc.DOC_TEST:
        doc_test(dc.LOG_FILE_NAME)
    else:
        input_test()
