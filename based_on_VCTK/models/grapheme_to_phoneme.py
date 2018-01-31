import tensorflow as tf
import numpy as np


class GraphemeToPhoneme(object):

    def __init__(self,
                 input_vocab_size, output_vocab_size,
                 end_token, model_parameters
                 ):
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.end_token = end_token
        self.model_parameters = model_parameters
        self.target_input_vocab_size = output_vocab_size + 1

    def build_train_operations(self,
                               characters, characters_seq_len,
                               phonemes, phonemes_seq_len,
                               train_parameters, reuse=None
                 ):
        with tf.variable_scope("encoder", reuse=reuse):
            _, encoder_states, _ = self.__build_encoder(
                characters, characters_seq_len, train_parameters["dropout_prob"]
            )
        with tf.variable_scope("decoder", reuse=reuse):
            decoder_cells, decoder_embedding = self.__build_decoder(
                train_parameters["dropout_prob"]
            )
            target_input = tf.concat(
                [tf.fill([tf.shape(phonemes)[0], 1], self.output_vocab_size), phonemes], 1
            )
            (decoder_rnn_output, decoder_output), _, _  = self.__buil_train_decoder(
                decoder_cells, decoder_embedding, encoder_states,
                target_input, phonemes_seq_len
            )
        loss = self.__loss(decoder_rnn_output, phonemes, phonemes_seq_len)
        tf.summary.scalar('loss', loss)

        global_step = tf.Variable(
            0, name='global_step', trainable=False
        )
        learning_rate = tf.train.exponential_decay(
            train_parameters["lr"], global_step,
            train_parameters["decay_steps"],
            train_parameters["decay_rate"]
        )
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients, variables = zip(*opt.compute_gradients(loss))
        train_op = opt.apply_gradients(zip(gradients, variables), global_step=global_step)

        summary = tf.summary.merge_all()
        #prediction=self.__build_prediction_decoder(
        #                           decoder_cells, decoder_embedding, encoder_states
        #                           )

        #return train_op, loss, global_step, summary, decoder_output,prediction
        return train_op, loss, global_step, summary, decoder_output

    def build_prediction(self,
                         characters, characters_seq_len, reuse=None
                         ):
        with tf.variable_scope("encoder", reuse=reuse):
            _, encoder_states, _ = self.__build_encoder(
                characters, characters_seq_len, 1.0
            )
        with tf.variable_scope("decoder", reuse=reuse):
            decoder_cells, decoder_embedding = self.__build_decoder(
                1.0
            )
            return self.__build_prediction_decoder(
                decoder_cells, decoder_embedding,
                encoder_states
            )

    def __build_encoder(self, X, X_seq_len, dropout_prob):
        #embedding = tf.get_variable(
        #    'embedding',
        #    shape=(self.input_vocab_size, self.model_parameters["embedding_size"]),
        #    dtype=tf.float32
        #)
        embedding = tf.Variable(np.identity(self.input_vocab_size,dtype=np.float32))

        embedded_inputs = tf.nn.embedding_lookup(embedding, X)
        cells_fw = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(self.model_parameters["num_units"]),
                output_keep_prob = dropout_prob
            )
            for i in xrange(self.model_parameters["num_layers"])
        ]
        cells_bw = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(self.model_parameters["num_units"]),
                output_keep_prob = dropout_prob
            )
            for i in xrange(self.model_parameters["num_layers"])
        ]
        return tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw, cells_bw, embedded_inputs,
            sequence_length=X_seq_len, dtype=tf.float32
        )

    def __build_decoder(self, dropout_prob):
        cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(self.model_parameters["num_units"]),
                output_keep_prob = dropout_prob
            ) for i in xrange(self.model_parameters["num_layers"])
        ]
        cells_net = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.MultiRNNCell(cells), self.target_input_vocab_size
        )
        #embedding = tf.get_variable(
        #    'embedding',
        #    shape=(
        #        self.target_input_vocab_size,
        #        self.model_parameters["embedding_size"]
        #    ), dtype=tf.float32
        #)
        embedding = tf.Variable(np.identity(self.target_input_vocab_size,dtype=np.float32))
        return cells_net, embedding

    def __buil_train_decoder(self, cells_net, embedding, states, Y, Y_seq_len):
        decoder_helper = tf.contrib.seq2seq.TrainingHelper(
            tf.nn.embedding_lookup(embedding, Y),
            Y_seq_len
        )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = cells_net,
            helper = decoder_helper,
            initial_state = states
        )
        return tf.contrib.seq2seq.dynamic_decode(training_decoder)

    def __loss(self, logits, Y, Y_seq_len):
        max_len = tf.reduce_max(Y_seq_len)
        masks = tf.sequence_mask(
            Y_seq_len, max_len, dtype=tf.float32
        )
        targets = tf.slice(Y, [0, 0], [-1, max_len])
        return tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets, weights=masks)

    def __build_prediction_decoder(self,
                                   cells_net, embedding, states
                                   ):
        start_token = self.output_vocab_size
        start_tokens = tf.tile(
            tf.constant([start_token], dtype=tf.int32),
            [tf.shape(states[0])[0]]
        )
        decoder_state = tuple([
            tf.contrib.seq2seq.tile_batch(
                state, multiplier=self.model_parameters["num_beams"]#A (possibly nested structure of) Tensor shaped [batch_size * multiplier, ...].
            ) for state in states
        ])
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cells_net, embedding, start_tokens, self.end_token,
            decoder_state, self.model_parameters["num_beams"]
        )
        (predictions, _, lengths) = tf.contrib.seq2seq.dynamic_decode(decoder)#(final_outputs, final_state, final_sequence_lengths).
        best_predictions = tf.slice(predictions.predicted_ids, [0, 0, 0], [-1, -1, 1])#.predicted_ids?
        #TypeError: Cannot convert a list containing a tensor of dtype <dtype: 'float32'> to <dtype: 'int32'> (Tensor is: <tf.Tensor 'decoder_1/decoder/transpose_1:0' shape=(?, ?, 1) dtype=float32>)
        #best_predictions = tf.slice(predictions, [0, 0, 0], [-1, -1, 1])
        best_predictions_lengths = tf.slice(lengths, [0, 0], [-1, 1])
        return best_predictions, best_predictions_lengths
