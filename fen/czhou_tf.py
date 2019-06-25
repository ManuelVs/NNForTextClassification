import tensorflow as tf


class ZhouCLSTMModel:
    '''
    Implementation proposal of: https://arxiv.org/pdf/1511.08630
    '''
    def __init__(self, embedding,
        conv_size    = 3,
        conv_filters = 300,
        drop_rate    = 0.5,
        lstm_units   = 150):
        '''Constructor.
        # Parameters:
        conv_size: Size of the convolutions. Number of words that takes each
            convolution step.
        conv_filters: Number of convolution filters.
        drop_rate: Drop rate for the final output of the LSTM layer.
        lstm_units: Size of the states of the LSTM layer.
        '''
        self._embedding    = embedding
        self._conv_size    = conv_size
        self._conv_filters = conv_filters
        self._drop_rate    = drop_rate
        self._lstm_units   = lstm_units

    def __call__(self, input):
        self._embedding_tf = self._create_embedding_layer(
            self._embedding, input)
        
        self._convolution_tf = self._create_convolutional_layers(
            self._conv_size,
            self._conv_filters,
            self._drop_rate,
            self._embedding_tf)
        
        self._lstm_tf = self._create_lstm_layer(
            self._lstm_units,
            self._convolution_tf)
        
        return self._lstm_tf

    def summary(self):
        print('embedding:', str(self._embedding_tf.shape))
        print('conv:', str(self._convolution_tf.shape))
        print('lstm:', str(self._lstm_tf.shape))

    def _create_embedding_layer(self, embedding, input_x):
        embedding = tf.Variable(initial_value=embedding)

        embedded_chars = tf.nn.embedding_lookup(
            embedding, tf.cast(input_x, 'int32'))

        return embedded_chars

    def _create_convolutional_layers(self,
        conv_size, num_filters, drop_rate, embedding):
        
        filter_height = conv_size
        filter_width = embedding.shape[2].value

        filter_shape = [filter_height, filter_width, 1, num_filters]

        W = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=filter_shape,
                stddev=0.1))
        b = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[num_filters]))

        embedding_expanded = tf.expand_dims(embedding, -1)
        conv = tf.nn.conv2d(
            input=embedding_expanded,
            filter=W,
            strides=[1,1,1,1],
            padding='VALID')
        conv_reduced = tf.reshape(
            tensor=conv,
            shape=[-1, conv.shape[1], conv.shape[3]])

        bias = tf.nn.bias_add(conv_reduced, b)
        c = tf.nn.relu(bias)

        d = tf.nn.dropout(c, rate=drop_rate)
        return d

    def _create_lstm_layer(self, lstm_units, conv_input):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_units)
        sequence = tf.unstack(conv_input, axis=1)
        (_, (h, _)) = tf.nn.static_rnn(lstm_cell, sequence, dtype=tf.float32)

        return h


if __name__ == '__main__':
    embedding_size  = 300
    num_words       = 1000
    sentence_length = 10

    embedding = [
        [float(i) for i in range(embedding_size)] for _ in range(num_words)
    ]
    data = [
        [i     for i in range(sentence_length)],
        [i + 1 for i in range(sentence_length)]
    ]

    model = ZhouCLSTMModel(embedding)
    model(data)
    model.summary()
