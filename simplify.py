""" Where training and simplification actually happen """

import sys
import tensorflow as tf
import time

import data_constants as dc
import data_utils
import simplifier_model as sm

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
        
        if count % dc.TRAIN_TEST_SPLIT == 0:
            valid_set.append([normal_ids, simple_ids])
        else:
            train_set.append([normal_ids, simple_ids])
        normal_line, simple_line = normal_f.readline(), simple_f.readline()
        count += 1

    normal_f.close()
    simple_f.close()

    return train_set, valid_set

def create_model(session, forward_only):
    """Create translation model"""
    
    normal_vocab_size, simple_vocab_size = 0, 0

    with open(dc.NORMAL_VOCAB_PATH, 'r+') as f:
        normal_vocab_size = len(f.readlines())
    with open(dc.SIMPLE_VOCAB_PATH, 'r+') as f:
        normal_vocab_size = len(f.readlines())

    model = sm.SimplifierModel(
                normal_vocab_size,
                simple_vocab_size, 
                dc.UNITS_PER_LAYER,
                dc.LAYERS,
                dc.MAX_GRAD_NORM,
                dc.BATCH_SIZE,
                dc.LEARNING_RATE,
                dc.LEARNING_RATE_DECAY_FACTOR,
                forward_only=forward_only,
                dtype=tf.float32)

    ckpt = tf.train.get_checkpoint_State(dc.CKPT_PATH)

    if skpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
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
        train, valid = read_data(dc.NORMAL_IDS_PATH, SIMPLE_IDS_PATH)
        train_size = len(data)
        valid_size = len(valid)

        print 'entering training loop'
        step_time, loss = 0.0, 0.0
        current_step = 0
        prev_losses = []
        while True:
            start_time = time.time()
            encoder_in, decoder_in, target_weights = model.get_batch(train)
            _, step_loss, _ = model.step(sess, encoder_in, decoder_in,
                                         target_weights, False)
            step_time += (time.time() - start_time) / dc.STEPS_PER_CHECKPOINT
            loss += step_loss / dc.STEPS_PER_CHECKPOINT
            current_step += 1

            if current_step % dc.STEPS_PER_CHECKPOINT == 0:
                print "loss: %d learnrate %d step-time: %d" % (
                        loss, model.learning_rate_eval(), step_time)
                
                if len(prev_losses) > 2 and loss > max(prev_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                checkpoint_path = os.path.join(dc.CKPT_PATH, 'simplify.ckpt')
                model.saver.save(sess, checkpoint_path,
                        global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                encoder_in, decoder_in, target_weights = model.get_batch(valid)
                _, val_loss, _ = model.step(sess, encoder_in, decoder_in,
                                            target_weights, True)

                print "validation loss: %d" % val_loss
            sys.stdout.flush()

def test():
    with tf.Session() as sess:
        model = create_model(sess, True)
        model.batch_size = 1
        
        normal_vocab, _ = data_utils.get_vocab(dc.NORMAL_VOCAB_PATH)
        _, rev_simple_vocab = data_utils.get_vocab(dc.SIMPLE_VOCAB_PATH)

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            token_sentence = data_utils.sentence_to_ids(sentence, normal_vocab)
            input_token_sentence = [(token_sentence, [])]
            encoder_in, decoder_in, target_weights = model.get_batch(
                    input_token_sentence)
            _, _, output_logits = model.step(sess, encoder_in, decoder_in,
                    target_weights, True)

            outputs = [int(np.argmax(logit, axis=1) for logit in output_logits)]

            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]

            print(" ".join([tf.compat.as_str(rev_simple_vocab) for output in
                            outputs]))

            print "> "
            sys.stdout.flush()
            sentence = sys.stdin.readline()

if __name__ == "__main__":
    train(dc.CREATE_DATA)
    test()
        

        


