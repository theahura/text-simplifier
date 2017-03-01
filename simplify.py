""" Where training and simplification actually happen """

import numpy as np
import os
import sys
import tensorflow as tf
import time

import data_constants as dc
import data_utils
import model as sm

def read_data(normal_path, simple_path):
    """Reads data from files and returns as list of tuples"""
    normal_f = open(normal_path, 'r+')
    simple_f = open(simple_path, 'r+')

    normal_line, simple_line  = normal_f.readline(), simple_f.readline()
    train_set = []
    valid_set = []
    count = 0

    while normal_line and simple_line:
        normal_ids = [int(x) for x in normal_line.split()]
        simple_ids = [int(x) for x in simple_line.split()]
        
        if count % dc.TRAIN_VALID_SPLIT == 0:
            valid_set.append([normal_ids, simple_ids])
        else:
            train_set.append([normal_ids, simple_ids])
        normal_line, simple_line = normal_f.readline(), simple_f.readline()
        count += 1

    normal_f.close()
    simple_f.close()

    return train_set, valid_set

def create_model(session, feed_previous):
    """Create translation model"""
    
    normal_vocab_size, simple_vocab_size = 0, 0

    with open(dc.NORMAL_VOCAB_PATH, 'r+') as f:
        normal_vocab_size = len(f.readlines())
    with open(dc.SIMPLE_VOCAB_PATH, 'r+') as f:
        simple_vocab_size = len(f.readlines())

    model = sm.SimplifierModel(
                normal_vocab_size,
                simple_vocab_size, 
                dc.UNITS_PER_LAYER,
                dc.LAYERS,
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

def train(create_data):
    """Trains a english to simple english translation model"""

    if create_data:
        data_utils.process_data()

    with tf.Session() as sess:
        print 'creating model'
        model = create_model(sess, False)

        print 'reading data'
        train, valid = read_data(dc.NORMAL_IDS_PATH, dc.SIMPLE_IDS_PATH)

        print 'entering training loop'
        step_time, loss = 0.0, 0.0
        current_step = 0
        prev_losses = []
        while current_step < dc.NUM_STEPS or dc.IGNORE_STEPS:
            start_time = time.time()
            encoder_in, decoder_in, target_weights = model.get_batch(train)
            _, step_loss, _ = model.step(sess, encoder_in, decoder_in,
                                         target_weights, False)
            step_time += (time.time() - start_time)/dc.STEPS_PER_CHECKPOINT

            loss += step_loss / dc.STEPS_PER_CHECKPOINT
            current_step += 1
            
            if current_step < dc.STEPS_PER_CHECKPOINT:
                print "Step: %f" % current_step
                print "Loss: %f" % step_loss
                print "Learning: %f" % model.learning_rate.eval()

            if current_step % dc.STEPS_PER_CHECKPOINT == 0:
                learnrate = model.learning_rate.eval()
                perplex = math.exp(float(loss)) if loss < 300 else float("inf")
                print "global step %d loss %f plex: %f learnrate %f step-time: %f" % (
                        model.global_step.eval(), loss, perplex, learnrate,
                        step_time)
                
                if len(prev_losses) > 2 and loss > max(
                        prev_losses[-1 * dc.DECAY_POINT:]):
                    sess.run(model.learning_rate_decay_op)
                prev_losses.append(loss)
                checkpoint_path = os.path.join(dc.CKPT_PATH, 'simplify.ckpt')
                model.saver.save(sess, checkpoint_path,
                        global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                encoder_in, decoder_in, target_weights = model.get_batch(valid)
                _, val_loss, outputs = model.step(sess, encoder_in, decoder_in,
                                                  target_weights, True)

                print "validation loss: %d" % val_loss
            sys.stdout.flush()

def test():
    with tf.Session() as sess:
        model = create_model(sess, True)
        model.batch_size = 1
        
        normal_vocab, _ = data_utils.get_vocabulary(dc.NORMAL_VOCAB_PATH)
        _, rev_simple_vocab = data_utils.get_vocabulary(dc.SIMPLE_VOCAB_PATH)

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            token_sentence = data_utils.sentence_to_ids(sentence, normal_vocab)

            print token_sentence

            padded_sentence = data_utils.pad_data(token_sentence, dc.MAX_LEN_IN,
                    True)

            print padded_sentence

            decoder = [dc.GO_ID] + [dc.EMPT_ID]*(dc.MAX_LEN_OUT - 1)

            input_token_sentence = [(padded_sentence.lower().split(), decoder)]
            encoder_in, decoder_in, target_weights = model.get_batch(
                    input_token_sentence)

            print encoder_in
            print decoder_in
            print target_weights
            _, loss, output_logits = model.step(sess, encoder_in, decoder_in,
                    target_weights, True)

            print "Loss: %f" % loss

            print len(output_logits[0])
            print output_logits[0]
            print np.max(output_logits[0], axis=1)

            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

            if dc.EOS_ID in outputs:
                outputs = outputs[:outputs.index(dc.EOS_ID)]

            print len(outputs)
            print outputs

            print(" ".join([tf.compat.as_str(rev_simple_vocab[output])
                for output in outputs]))

            print "> "
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def self_test():
  """Test the model."""
  with tf.Session() as sess:
    print("Pipeline test for neural translation model")
    # Create model with vocabularies of 10, 2 layers of 32, batch size 32.
    dc.MAX_LEN_IN = 2
    dc.MAX_LEN_OUT = 2
    model = sm.SimplifierModel(10, 10, 32, 2, 5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.global_variables_initializer())

    data_set = ([([1, 7], [2, 2]), ([8, 3], [4, 4]), ([9, 5], [6, 6])])
    for _ in xrange(100):  # Train the fake model for 50 steps.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
              target_weights, False)
      print "Loss %f" % step_loss
      
    model.batch_size = 1
    input_token = [([1, 7], [0, 0])]
    encode, decode, weights = model.get_batch(input_token)
    _, _, output_logits = model.step(sess, encode, decode, weights, True)
    print output_logits
    print len(output_logits)
    print len(output_logits[0])
    print np.argmax(output_logits[0], axis=1)
    outputs = [int(np.argmax(logit, axis=1)[0]) for logit in output_logits]
    print len(outputs)
    print outputs

    input_token = [([9, 5], [0, 0])]
    encode, decoder, weights = model.get_batch(input_token)
    _, _, output_logits = model.step(sess, encode, decode, weights, True)
    outputs = [int(np.argmax(logit, axis=1)[0]) for logit in output_logits]
    print len(outputs)
    print outputs



if __name__ == "__main__":
    if dc.SELF_TEST:
        self_test()
    else:
        if dc.TRAIN:
            train(dc.CREATE_DATA)

        test()
