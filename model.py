"""
Author: Amol Kapoor

Description: Model for seq2seq text simplifier
"""

import numpy as np
import tensorflow as tf
import data_utils

class SimplifierModel():

    def __init__(self,
                 normal_vocab_size,
                 simple_vocab_size,
                 units_per_layer,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 num_samples=512,
                 feed_previous=False,
                 dtype=tf.float32):
        """Inits the model"""

        self.normal_vocab_size = normal_vocab_size
        self.simple_vocab_size = simple_vocab_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
                float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        output_projection = None
        softmax_loss_function = None

        # Add sampleloss here
        # Add conv net layers here

        def single_cell():
            return tf.contrib.rnn.BasicLSTMCell(size)

        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in
                range(num_layers)])
        else:
            cell = single_cell()

        def seq2seq_f(encoder_inputs, decoder_inputs, feed_previous):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    cell,
                    num_encoder_symbols=normal_vocab_size,
                    num_decoder_symbols=simple_vocab_size,
                    embedding_size=size,
                    feed_previous=feed_previous,
                    dtype=dtype)
       

        self.encoder_inputs = []
        self.decoder_inputs = [] 
        self.target_weights = []
        for i in xrange(dc.MAX_LEN_IN):
            self.enocder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                name="encoder%d" % i))
        for i in xrange(dc.MAX_LEN_OUT + 1): # Not sure if +1 is needed.
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                name="decoder%d" % i))
            self.target_weights.append(tf.placeholder(dtype, shape=[None],
                name="weights%d" % i))

        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]
        if feed_previous: 
            self.out, self.loss = tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, [(dc.MAX_LEN_IN, dc.MAX_LEN_OUT)], 
                    lambda x, y: seq2seq_f(x, y, True), 
                    softmax_loss_function=softmax_loss_function)
            # Add output projection stuff
        else:
            self.out, self.loss = tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, [(dc.MAX_LEN_IN, dc.MAX_LEN_OUT)], 
                    lambda x, y: seq2seq_f(x, y, False), 
                    softmax_loss_function=softmax_loss_function)

        params = tf.trainable_variables()
        if not feed_previous:
            opt = tf.train.AdamOptimizer(self.learning_rate)

            gradients = tf.gradients(self.loss[0], params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)

            self.grad_norm = norm
            self.update = opt.apply_gradients(zip(clipped_gradients, params),
                    global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             feed_previous):
        """Runs a step of the model"""
        encoder_size, decoder_size = dc.MAX_LEN_IN, dc.MAX_LEN_OUT

        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder sizes unequal")
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder sizes unequal")
        if len(target_weights) != decoder_size:
            raise ValueError("Target sizes unequal")

        input_feed = {}

        for i in xrange(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in xrange(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.target_weights[i].name] = target_weights[i]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not feed_previous:
            output_feed = [self.update,
                           self.grad_norm,
                           self.loss]    
        else:
            output_feed = [self.loss]
            for i in xrange(decoder_size):
                output_feed.append(self.out[i])

        outputs = session.run(output_feed, input_feed) 

        if feed_previous:
            return None, outputs[0], outputs[1:]
        else:
            return outputs[1], outputs[2], None

    def get_batch(self, data):
        """Gets the next batch from the data set"""
        encoder_size, decoder_size = dc.MAX_LEN_IN, MAX_LEN_OUT
        encoder_inputs, decoder_inputs = [], []

        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data)
            encoder_inputs.appened(encoder_input)
            decoder_inputs.append(decoder_input)

        batch_encoder_in, batch_decoder_in, batch_weights = [], [], []

        for sentence_location in xrange(encoder_size):
            batch_encoder_in.append(
                    np.array([encoder_inputs[batch_id][sentence_location]
                              for batch_id in xrange(self.batch_size)],
                              dtype=np.int32))

        for sentence_location in xrange(decoder_size):
            batch_decoder_in.append(
                    np.array([decoder_inputs[batch_id][sentence_location]
                              for batch_id in xrange(self.batch_size)],
                              dtype=np.int32))

            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_id in xrange(self.batch_size):
                if sentence_location < decoder_size - 1:
                    target = decoder_inputs[batch_id][sentence_location + 1]
                if sentence_location == decoder_size - 1 or target == dc.PAD_ID:
                    batch_weight[batch_id] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_in, batch_decoder_in, batch_weights
