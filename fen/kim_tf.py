import tensorflow as tf


class KimConvolutionalModel:
    '''
    Implementation proposal of: https://arxiv.org/pdf/1408.5882.pdf
    '''
    def __init__(self,
        embeddings_configuration,
        conv_configurations = [(3, 100), (4, 100), (5, 100)],
        drop_rate           = 0.5):
        '''Constructor.
        # Parameters:
        embeddings: List of embeddings configuration. Each configuration is a
            pair of the form (embedding, trainable). `embedding` is a numpy
            array and `trainable` is a boolean that indicates whether that
            embedding is trainable or not.
        conv_configurations: List of pairs. Each pair represents a
            convolution configuration. Each configuration determines the
            size and number of each filter.
        '''

        self._embeddings_configuration = embeddings_configuration
        self._conv_configurations = conv_configurations
        self._drop_rate = drop_rate

    def __call__(self, input):
        self._embeddings_tf = tf.stack(
            values = [
                self._create_embedding_layer(e, input)
                for e in self._embeddings_configuration],
            axis = 1
        )

        self._convolutions_tf = self._create_convolutional_layers(
            self._conv_configurations, self._embeddings_tf)
        
        self._add_tf = self._create_add_layers(self._convolutions_tf)

        self._poolings_tf = self._create_maxpooling_layer(
            self._add_tf)

        self._reshape_tf = self._create_reshape_layer(self._poolings_tf)
        self._dropout_tf = tf.nn.dropout(
            self._reshape_tf,
            rate = self._drop_rate)

        return self._dropout_tf

    def summary(self):
        print('embedding:', str(self._embeddings_tf.shape))
        for c in self._convolutions_tf:
            print('conv:', str(c.shape))
        for a in self._add_tf:
            print('add:', str(a.shape))
        for p in self._poolings_tf:
            print('pool:', str(p.shape))
        print('reshape:', str(self._reshape_tf.shape))

    def _create_embedding_layer(self, embedding_configuration, input_x):
        return tf.nn.embedding_lookup(
            params = tf.Variable(
                initial_value = embedding_configuration[0],
                trainable     = embedding_configuration[1]),
            ids = tf.cast(input_x, 'int32')
        )

    def _create_convolutional_layers(self, configuration, input_embedding):
        '''Creates the convolutional layers.
        # Parameters:
        configuration: A list. It must be of the form
            [(filter_size, num_filters), ...]
        # Returns:
        A list of tensorflow nodes. Each node 'i' computes the configuration 'i'.
        '''
        convolutions = []
        for filter_height, num_filters in configuration:
            filter_width = input_embedding.shape[3].value
            filter_shape = [1, filter_height, filter_width, num_filters]

            # Create weights and bias
            W = tf.Variable(
                initial_value=tf.truncated_normal(
                    shape=filter_shape,
                    stddev=0.1))
            b = tf.Variable(
                initial_value=tf.truncated_normal(
                    shape=[num_filters]))

            conv = tf.nn.conv2d(
                input=input_embedding,
                filter=W,
                strides=[1, 1, 1, 1],
                padding="VALID")
            bias = tf.nn.bias_add(conv, b)
            h = tf.nn.relu(bias)
            convolutions.append(h)

        return convolutions

    def _create_add_layers(self, convolutions):
        return [
            tf.reduce_sum(
                input_tensor = c,
                axis=1,
                keepdims=True)
            for c in convolutions
        ]

    def _create_maxpooling_layer(self, tensors):
        '''Creates the maxpooling layer. Computes maxpooling on each node
        # Parameters:
        input_convolutions: List of tensorflow nodes.
        # Returns:
        A list of tensorflow nodes. Each node 'i' computes the maxpooling of node 'i'
        '''
        return [
            tf.reshape(
                tensor = tf.nn.max_pool(
                    value=t,
                    ksize=[1, 1, t.shape[2], 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID'),
                shape = [-1, t.shape[3]]
            )
            for t in tensors
        ]

    def _create_reshape_layer(self, tensors):
        '''Creates a flatten layer
        '''
        return tf.concat(tensors, axis=1)


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

    model = KimConvolutionalModel([
        (embedding, True), (embedding, False)
    ])
    model(data)
    model.summary()
