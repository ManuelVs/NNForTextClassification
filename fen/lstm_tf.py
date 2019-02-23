import tensorflow as tf

class LSTMModel:

    def __init__(self, sentence_length, embedding, num_classes):
        self.sentence_length = sentence_length
        self.num_classes = num_classes

        self.input = tf.placeholder(
            dtype='int32',
            shape=(None, sentence_length)
        )

        embedding_tf   = self._create_embedding_layer(embedding, self.input)
        lstm_tf        = self._create_lstm_layer(embedding_tf)
        dense_tf       = self._create_dense_layer(num_classes, lstm_tf)
        self.output    = dense_tf

        print('input    : ' + str(self.input.shape))
        print('embedding: ' + str(embedding_tf.shape))
        print('lstm     : ' + str(lstm_tf.shape))
        print('dense    : ' + str(dense_tf.shape))


    def _create_embedding_layer(self, embedding_array, input_x):
        embedding = tf.Variable(
            initial_value=embedding_array)
        
        embedded_chars = tf.nn.embedding_lookup(embedding, input_x)
        
        return embedded_chars


    def _create_lstm_layer(self, embedding_input):
        lstm_cell = tf.contrib.rnn.LSTMCell(10)

        sequence = tf.unstack(embedding_input, axis=1)

        (_, (h, _)) = tf.nn.static_rnn(lstm_cell, sequence, dtype=tf.float32)

        return h


    def _create_dense_layer(self, num_classes, lstm_input):
        input_size = lstm_input.shape[1].value
        W = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[input_size, num_classes],
                stddev=0.1))
        b = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[num_classes]))

        dense = tf.nn.xw_plus_b(lstm_input, W, b)

        return dense
