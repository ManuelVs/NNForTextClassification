import tensorflow as tf

class KimConvolutionalModel:

    '''Constructor.
    Parameters:
      sentence_length: Length of all sentences
      embedding: numpy array representing the embedding. The first row (word) must be 0
    '''
    def __init__(self, sentence_length, embedding, num_classes):
        conv_configurations = [(3, 100), (4, 100), (5, 100)]

        self.input      = self._create_input(sentence_length)
        embedding_tf    = self._create_embedding_layer(embedding, self.input)
        convolutions_tf = self._create_convolutional_layers(conv_configurations, embedding_tf)
        poolings_tf     = self._create_maxpooling_layer(convolutions_tf)
        concatenate_tf  = self._create_concatenate_layer(poolings_tf)
        flatten_tf      = self._create_flatten_layer(concatenate_tf)
        dense_tf        = self._create_dense_layer(num_classes, flatten_tf)
        self.output     = dense_tf
    
        print('input    : ' + str(self.input.shape))
        print('embedding: ' + str(embedding_tf.shape))
        for c in convolutions_tf:
            print('conv     : ' + str(c.shape))
        for p in poolings_tf:
            print('pool     : ' + str(p.shape))
        print('concat   : ' + str(concatenate_tf.shape))
        print('flatten  : ' + str(flatten_tf.shape))
        print('output   : ' + str(dense_tf.shape))

    '''Creates the input.
    Parameters:
      sentence_length: Length of all sentences
    Returns:
      A Tensorflow node, represents the input
    '''
    def _create_input(self, sentence_length):
        return tf.placeholder(tf.int32, [None, sentence_length])

    '''Creates the embedding.
    Parameters:
      embedding_array: Array representing the embedding, used for initialization. Numpy array or Tensorflow tensor
      input_x: Preceding tensorflow node. It should be the sentence(s) represented with word index
    Returns:
      Tensorflow node that computes the new representation
    '''
    def _create_embedding_layer(self, embedding_array, input_x):
        embedding = tf.Variable(
            initial_value=embedding_array,
            name="embedding")
        
        embedded_chars = tf.nn.embedding_lookup(embedding, input_x)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        return embedded_chars_expanded
    
    '''Creates the convolutional layers.
    Parameters:
      configuration: A list. It must be of the form [(filter_size, num_filters), ...]
    Returns:
      A list of tensorflow nodes. Each node 'i' computes the configuration 'i'.
    '''
    def _create_convolutional_layers(self, configuration, input_embedding):
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

    '''Creates the maxpooling layer. Computes maxpooling on each node
    Parameters:
      input_convolutions: List of tensorflow nodes.
    Returns:
      A list of tensorflow nodes. Each node 'i' computes the maxpooling of node 'i'
    '''
    def _create_maxpooling_layer(self, input_convolutions):
        pooling = []
        for conv in input_convolutions:
            ksize=[1, conv.shape[1].value, 1, 1]
            pooled = tf.nn.max_pool(
                value=conv,
                ksize=ksize,
                strides=[1, 1, 1, 1],
                padding='VALID')
            
            pooling.append(pooled)
        
        return pooling
    
    '''Creates the concatenation layer. Computes the concatenations of all tensors
    Parameters:
      input_poolings: List of tensorflow nodes.
    Returns:
      A tensorflow node that computes the concatenations of all tensors
    '''
    def _create_concatenate_layer(self, input_poolings):
        conc = tf.concat(
            values=input_poolings,
            axis=3)
        
        return conc

    '''Creates a flatten layer
    '''
    def _create_flatten_layer(self, concatenate_input):
        num_filters_total = concatenate_input.shape[3].value
        flat = tf.reshape(concatenate_input, [-1, num_filters_total])
        return flat

    '''Creates a dense layer
    '''
    def _create_dense_layer(self, num_classes, flatten_input):
        input_size = flatten_input.shape[1].value
        W = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[input_size, num_classes],
                stddev=0.1))
        b = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[num_classes]))

        dense = tf.nn.xw_plus_b(flatten_input, W, b)

        return dense

