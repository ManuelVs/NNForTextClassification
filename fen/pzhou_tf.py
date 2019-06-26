import tensorflow as tf


class ZhouBLSTMCNNModel:
    '''
    Implementation proposal of: https://arxiv.org/pdf/1611.06639v1.pdf
    '''
    def __init__(self,
        embedding,
        em_drop_rate = 0.5,
        lstm_units   = 300,
        lstm_drop_rate = 0.2,
        conv_size    = (3, 3),
        conv_filters = 100,
        pool_size    = (2, 2),
        pool_drop_rate = 0.4):
        '''Constructor.
        # Parameters:
        embedding: Numpy array representing the embedding.
        em_drop_rate: Drop rate after the embedding layer.
        lstm_units: Size of the internal states of the LSTM cells.
        lstm_drop_rate: Drop rate after the lstm layer.
        conv_size: Size of the convolutions.
        conv_filters: Number of convolutions filters.
        pool_size: Size for the max pooling layer.
        pool_drop_rate: Drop rate of the max pooling layer.
        '''
        self._embedding      = embedding
        self._em_drop_rate   = em_drop_rate
        self._lstm_units     = lstm_units
        self._lstm_drop_rate = lstm_drop_rate
        self._conv_size      = conv_size
        self._conv_filters   = conv_filters
        self._pool_size      = pool_size
        self._pool_drop_rate = pool_drop_rate

    def __call__(self, input):
        self._embedding_tf = self._create_embedding_layer(
            self._em_drop_rate, self._embedding, input)

        self._sequences_tf = self._create_blstm_layer(
            self._lstm_units,
            self._lstm_drop_rate,
            self._embedding_tf)

        self._convolution_tf = self._create_convolutional_layer(
            self._conv_size,
            self._conv_filters,
            self._sequences_tf)
        self._pooling_tf = self._create_maxpooling_layer(
            self._pool_size,
            self._pool_drop_rate,
            self._convolution_tf)

        self._flatten_tf = self._create_flatten_layer(self._pooling_tf)

        return self._flatten_tf

    def summary(self):
        print("embedding: " + str(self._embedding_tf.shape))
        print("lstm: " + str(self._sequences_tf.shape))
        print("conv: " + str(self._convolution_tf.shape))
        print("pooling: " + str(self._pooling_tf.shape))
        print("flatten: " + str(self._flatten_tf.shape))

    def _create_embedding_layer(self, em_drop_rate, embedding, input_x):
        embedding = tf.Variable(initial_value=embedding)

        embedded_chars = tf.nn.embedding_lookup(
            embedding, tf.cast(input_x, 'int32'))

        return tf.nn.dropout(embedded_chars, rate=em_drop_rate)

    def _create_blstm_layer(self, lstm_units, lstm_drop_rate, embedding):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_units)
        sequence = tf.unstack(embedding, axis=1)
        hs, _, _ = tf.nn.static_bidirectional_rnn(lstm_cell, lstm_cell,
            sequence,
            dtype=tf.float32)
        
        hs = tf.stack(
            values=hs,
            axis=1)
        ss = tf.math.reduce_sum(
            tf.reshape(hs, shape=[-1, hs.shape[1], 2, lstm_units]),
            axis=2
        )

        return tf.nn.dropout(ss, rate=lstm_drop_rate)

    def _create_convolutional_layer(self,
        conv_size, num_filters, tensor):
        
        print(str(tensor.shape))

        filter_heigth = conv_size[0]
        filter_width  = conv_size[1]

        filter_shape = [filter_heigth, filter_width,
            1, num_filters]

        W = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=filter_shape,
                stddev=0.1))
        b = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[num_filters]))

        tensor_expanded = tf.expand_dims(tensor, -1)
        conv = tf.nn.conv2d(
            input=tensor_expanded,
            filter=W,
            strides=[1,1,1,1],
            padding='VALID')

        bias = tf.nn.bias_add(conv, b)
        c = tf.nn.relu(bias)

        return c

    def _create_maxpooling_layer(self, size, pool_drop_rate, conv):
        pooled = tf.nn.max_pool3d(
            input=tf.expand_dims(conv, -1),
            ksize=[1, size[0], size[1], conv.shape[3], 1],
            strides=[1, size[0], size[1], conv.shape[3], 1],
            padding='VALID')
        
        return tf.nn.dropout(pooled, rate=pool_drop_rate)

    def _create_flatten_layer(self, tensor):
        return tf.reshape(tensor, [-1, tensor.shape[1] * tensor.shape[2]])


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

    model = ZhouBLSTMCNNModel(embedding)
    model(data)
    model.summary()
