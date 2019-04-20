import tensorflow as tf


class KimConvolutionalModel:

    def __init__(self,
                 embedding,
                 conv_configurations=[(3, 100), (4, 100), (5, 100)]):
        '''Constructor.
        Parameters:
          sentence_length: Length of all sentences
          embedding: numpy array representing the embedding. The first row (word) must be 0
        '''

        self._embedding = embedding
        self._conv_configurations = conv_configurations

    def __call__(self, input):
        self._embedding_tf = self._create_embedding_layer(
            self._embedding, input)
        self._convolutions_tf = self._create_convolutional_layers(
            self._conv_configurations, self._embedding_tf)
        self._poolings_tf = self._create_maxpooling_layer(
            self._convolutions_tf)
        self._concatenate_tf = self._create_concatenate_layer(
            self._poolings_tf)
        self._flatten_tf = self._create_flatten_layer(self._concatenate_tf)

        return self._flatten_tf

    def summary(self):
        print('embedding:', str(self._embedding_tf.shape))
        for c in self._convolutions_tf:
            print('conv:', str(c.shape))
        for p in self._poolings_tf:
            print('pool:', str(p.shape))
        print('concat:', str(self._concatenate_tf.shape))
        print('features:', str(self._flatten_tf.shape))

    def _create_embedding_layer(self, embedding_array, input_x):
        '''Creates the embedding.
        Parameters:
        embedding_array: Array representing the embedding, used for initialization. Numpy array or Tensorflow tensor
        input_x: Preceding tensorflow node. It should be the sentence(s) represented with word index
        Returns:
        Tensorflow node that computes the new representation
        '''
        embedding = tf.Variable(
            initial_value=embedding_array,
            name="embedding")

        embedded_chars = tf.nn.embedding_lookup(
            embedding, tf.cast(input_x, 'int32'))
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        return embedded_chars_expanded

    def _create_convolutional_layers(self, configuration, input_embedding):
        '''Creates the convolutional layers.
        Parameters:
        configuration: A list. It must be of the form [(filter_size, num_filters), ...]
        Returns:
        A list of tensorflow nodes. Each node 'i' computes the configuration 'i'.
        '''
        convolutions = []
        for (filter_height, num_filters) in configuration:
            filter_width = input_embedding.shape[2].value
            filter_shape = [filter_height, filter_width, 1, num_filters]

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

    def _create_maxpooling_layer(self, input_convolutions):
        '''Creates the maxpooling layer. Computes maxpooling on each node
        Parameters:
        input_convolutions: List of tensorflow nodes.
        Returns:
        A list of tensorflow nodes. Each node 'i' computes the maxpooling of node 'i'
        '''
        pooling = []
        for conv in input_convolutions:
            ksize = [1, conv.shape[1].value, 1, 1]
            pooled = tf.nn.max_pool(
                value=conv,
                ksize=ksize,
                strides=[1, 1, 1, 1],
                padding='VALID')

            pooling.append(pooled)

        return pooling

    def _create_concatenate_layer(self, input_poolings):
        '''Creates the concatenation layer. Computes the concatenations of all tensors
        Parameters:
        input_poolings: List of tensorflow nodes.
        Returns:
        A tensorflow node that computes the concatenations of all tensors
        '''
        conc = tf.concat(
            values=input_poolings,
            axis=3)

        return conc

    def _create_flatten_layer(self, concatenate_input):
        '''Creates a flatten layer
        '''
        num_filters_total = concatenate_input.shape[3].value
        flat = tf.reshape(concatenate_input, [-1, num_filters_total])
        return flat
