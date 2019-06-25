import tensorflow as tf


class ZhangDependencyModel:
    '''
    Implementation proposal of: https://arxiv.org/pdf/1611.02361.pdf
    '''
    def __init__(self, embeddings,
        conv_size    = 20,
        conv_filters = 100):
        '''Constructor.
        # Parameters:
        embeddings: List of numpy arrays representing the embeddings.
        conv_size: Size of the convolutions.
        conv_filters: Number of convolutions filters.
        '''
        self._embeddings   = embeddings
        self._conv_size    = conv_size
        self._conv_filters = conv_filters

    def __call__(self, input):
        self._embeddings_tf = [
            self._create_embedding_layer(embedding, input)
            for embedding in self._embeddings
        ]

        self._sequences_tf = [
            self._create_lstm_layers(embedding, 'lstm' + str(i))
            for i, embedding in enumerate(self._embeddings_tf)
        ]

        self._reshaped = self._reshape(self._sequences_tf)

        self._convolution = self._create_convolutional_layer(
            self._conv_size,
            self._conv_filters,
            self._reshaped)
        self._pooling = self._create_maxpooling_layer(self._convolution)

        return self._pooling

    def summary(self):
        for e in self._embeddings_tf:
            print("embedding: " + str(e.shape))
        print("lstm: " + str(self._reshaped.shape))
        print("conv: " + str(self._convolution.shape))
        print("pooling: " + str(self._pooling.shape))
        pass

    def _create_embedding_layer(self, embedding, input_x):
        embedding = tf.Variable(initial_value=embedding)

        embedded_chars = tf.nn.embedding_lookup(
            embedding, tf.cast(input_x, 'int32'))

        return embedded_chars

    def _create_lstm_layers(self, embedding, scope):
        lstm_units = embedding.shape[2]

        lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_units)
        sequence = tf.unstack(embedding, axis=1)
        hs, _ = tf.nn.static_rnn(lstm_cell, sequence,
            dtype=tf.float32,
            scope=scope)
        
        return hs

    def _reshape(self, tensors):
        # After stack, the dimensions are:
        # [
        #   num_embeddings,
        #   input_length,
        #   batch_size,
        #   embedding_size
        # ]
        # And we want it to be:
        # [
        #   batch_size,
        #   num_embeddings,
        #   input_length,
        #   embedding_size
        # ]
        return tf.transpose(
            tf.stack(tensors),
            perm=[2,0,1,3])

    def _create_convolutional_layer(self,
        conv_size, num_filters, tensor):
        
        filter_depth  = tensor.shape[1].value
        filter_heigth = tensor.shape[2].value
        filter_width  = conv_size

        filter_shape = [filter_depth, filter_heigth, filter_width,
            1, num_filters]

        W = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=filter_shape,
                stddev=0.1))
        b = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[num_filters]))

        tensor_expanded = tf.pad(
            tensor,
            paddings = [
                [0, 0], # 0 padding on dimension 0 (batch)
                [0, 0], # 0 padding on dimension 1 (num_embeddings)
                [0, 0], # 0 padding on dimension 2 (embedding_size)
                [conv_size - 1, conv_size - 1]
            ],
            mode = "CONSTANT",
            constant_values=0
        )
        tensor_expanded = tf.expand_dims(tensor_expanded, axis=4)

        conv = tf.nn.conv3d(
            input=tensor_expanded,
            filter=W,
            strides=[1,1,1,1,1],
            padding='VALID')

        bias = tf.nn.bias_add(conv, b)
        c = tf.nn.relu(bias)

        return tf.reshape(c, shape=[-1, c.shape[3], c.shape[4], 1])

    def _create_maxpooling_layer(self, conv):
        pooled = tf.nn.max_pool(
            value=conv,
            ksize=[1, conv.shape[1].value, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID')
        
        return tf.reshape(pooled, shape=[-1, pooled.shape[2]])


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

    model = ZhangDependencyModel([embedding, embedding])
    model(data)
    model.summary()
