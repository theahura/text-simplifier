"""Model for seq2seq text simplifier. Influenced by tf seq2seq tutorial."""

import numpy as np
import random
import tensorflow as tf

import data_constants as dc
import data_utils

class SimplifierModel():

    def __init__(self,
                 normal_vocab_size,
                 simple_vocab_size,
                 units_per_layer,
                 max_grad_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 num_samples=512,
                 feed_previous=False,
                 dtype=tf.float32):
        """Inits the model. Set feed_previous to true for training, as it
        defines whether the predicted output word is fed into the model at
        timestep t+1, or if the given translation is."""

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

        # See Jean et. al. Implementation taken from tf tutorial.
        if (num_samples > 0 and num_samples < self.simple_vocab_size and 
                dc.USE_SAMPLED_SOFTMAX):
            print "Using sampled softmax."
            w_t = tf.get_variable('proj_w', [self.simple_vocab_size,
                units_per_layer], dtype=dtype)
            w = tf.transpose(w_t) 
            b = tf.get_variable('proj_b', [self.simple_vocab_size], dtype=dtype)
            output_projection = (w, b)

            if dc.DEBUG:
                print "OUTPUT PROJECTION SET"

            def sampled_loss(labels, inputs):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.cast(
                        tf.nn.sampled_softmax_loss(
                            weights=local_w_t,
                            biases=local_b,
                            labels=labels,
                            inputs=local_inputs,
                            num_sampled=num_samples,
                            num_classes=self.simple_vocab_size),
                        dtype)
            softmax_loss_function = sampled_loss
        else:
            print "Not using sampled softmax"
              
        # Use base LSTM with dropout with multiple layers.
        def single_cell(units):
            cell = tf.contrib.rnn.LSTMCell(units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, dc.INPUT_KEEP_PROB,
                    dc.OUTPUT_KEEP_PROB)
            return cell 

        cell = single_cell(dc.LAYER_SETUP[0])
        if len(dc.LAYER_SETUP) > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell(units) for units in
                dc.LAYER_SETUP])

        self.encoder_inputs = []
        self.decoder_inputs = [] 
        self.target_weights = []
        for i in xrange(dc.MAX_LEN_IN):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                name="encoder%d" % i))
        for i in xrange(dc.MAX_LEN_OUT + 1): 
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                name="decoder%d" % i))
            self.target_weights.append(tf.placeholder(dtype, shape=[None],
                name="weights%d" % i))

        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        if dc.DEBUG:
            print "Normal size %d Simple size %d" % (normal_vocab_size,
                    simple_vocab_size)

        # Envoke attention model.
        self.out, state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    self.encoder_inputs[:dc.MAX_LEN_IN],
                    self.decoder_inputs[:dc.MAX_LEN_OUT],
                    cell,
                    num_encoder_symbols=normal_vocab_size,
                    num_decoder_symbols=simple_vocab_size,
                    embedding_size=units_per_layer,
                    output_projection=output_projection,
                    feed_previous=feed_previous,
                    dtype=dtype)

        if dc.DEBUG:
            self.state = state
            print "Output len %d" % len(self.out)
            print "Tensor len"
            print self.out[0].get_shape()
            print "Target len %d" % len(targets)
            print "Target weights len %d" % len(self.target_weights)

        # L2 regularization done on LSTM cell weights.
        l2 = dc.REG_CONST * sum(tf.nn.l2_loss(x) for x in
                tf.trainable_variables() if ('lstm_cell' in x.name and not
                    'biases' in x.name))

        self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.out,
                targets[:dc.MAX_LEN_OUT],
                self.target_weights[:dc.MAX_LEN_OUT],
                softmax_loss_function=softmax_loss_function) + l2

        if dc.DEBUG:
            print "Loss len"
            print self.loss.get_shape()

        # Need to map back from sampled subspace to vocab subspace.
        self.out_proj = None
        if output_projection is not None:
            if dc.DEBUG:
                print "Output Projection shape"
                print output_projection[0].get_shape()
            self.out_proj = [tf.matmul(output, output_projection[0]) +
                output_projection[1] for output in self.out]

        params = tf.trainable_variables()

        if dc.DEBUG:
            print 'PARAMETERS'
            print [param.name for param in params]
            print len(params)

        # When trianing, clips norm if hyperparameter is set appropriately.
        if not feed_previous:
            if max_grad_norm:
                print "Using clipping"
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)

                gradients = tf.gradients(self.loss, params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_grad_norm)

                self.grad_norm = norm
                self.update = opt.apply_gradients(zip(clipped_gradients, params),
                        global_step=self.global_step)
            else:
                print "Not using clipping"
                self.update = tf.train.GradientDescentOptimizer(
                        self.learning_rate).minimize(self.loss,
                                self.global_step)

        if dc.DEBUG:
            print "Var count: %d" % len(tf.global_variables())
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

        # Construct inputs for the graph.
        input_feed = {}

        for i in xrange(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in xrange(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.target_weights[i].name] = target_weights[i]

        last_target = self.decoder_inputs[decoder_size].name

        # The target words are all offset by one. Need an ending that is set to
        # all zeros as a result.
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if dc.DEBUG:
            print "INPUT FEED:"
            print input_feed

        # Construct output actions for the graph to run.
        if not feed_previous:
            output_feed = [self.update,
                           self.loss]    
            if dc.DEBUG:
                output_feed.append(self.state)
        else:
            output_feed = [self.loss]
            for i in xrange(decoder_size):
                if self.out_proj:
                    output_feed.append(self.out_proj[i])
                else:
                    output_feed.append(self.out[i])

        # Run step.
        outputs = session.run(output_feed, input_feed) 

        if dc.DEBUG and not feed_previous:
            print "STATES"
            print len(outputs[2])
            print outputs[2]

        # Based on whether we are in training or testing mode, output different
        # things.
        if feed_previous:
            return outputs[0], outputs[1:]
        else:
            return outputs[1], None

    def get_batch(self, data):
        """Gets the next batch from the data set"""
        encoder_size, decoder_size = dc.MAX_LEN_IN, dc.MAX_LEN_OUT
        encoder_inputs, decoder_inputs = [], []

        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data)
            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)

        batch_encoder_in, batch_decoder_in, batch_weights = [], [], []

        # Input sentences need to be transposed from aligned horizontally to
        # vertically, such that each word in a batch is aligned with all of the
        # other words at a certain input position. Heavily influenced from tf
        # tutorial.
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
                if (sentence_location == decoder_size - 1 or target ==
                        dc.EMPT_ID):
                    batch_weight[batch_id] = 0.0
            batch_weights.append(batch_weight)

        if dc.DEBUG:
            print "ENCODER IN"
            print batch_encoder_in
            print "DECODER IN"
            print batch_decoder_in
            print "WEIGHTS"
            print batch_weights

        return batch_encoder_in, batch_decoder_in, batch_weights
