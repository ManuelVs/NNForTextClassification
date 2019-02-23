import tensorflow as tf

class ShinContextualModel:

    '''Constructor.
    Parameters:
      sentence_length: Length of all sentences
      embedding: numpy array representing the embedding. The first row (word) must be 0
    '''
    def __init__(self, sentence_length, embedding, num_classes, conv_size = 3, num_layers = 2):
        self.input     = tf.placeholder('int32', [None, sentence_length])

        embedding_tf   = self._create_embedding_layer(embedding, self.input)
        convolution_tf = self._create_convolutional_layers(conv_size, num_layers, embedding_tf)
        pooling_tf     = self._create_maxpooling_layer(convolution_tf)
        dense_tf       = self._create_dense_layer(num_classes, pooling_tf)
        self.output    = dense_tf

        print('input    : ' + str(self.input.shape))
        print('embedding: ' + str(embedding_tf.shape))
        print('conv     : ' + str(convolution_tf.shape))
        print('pool     : ' + str(pooling_tf.shape))
        print('output   : ' + str(self.output.shape))


    def _create_embedding_layer(self, embedding, input_x):
        embedding = tf.Variable(initial_value=embedding)
        
        embedded_chars = tf.nn.embedding_lookup(embedding, input_x)

        return embedded_chars
    

    def _create_convolutional_layers(self, conv_size, num_layers, embedding):
        filter_height = conv_size
        filter_width  = embedding.shape[2].value

        filter_shape = [filter_height, filter_width, filter_width]

        W = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=filter_shape,
                stddev=0.1))
        b = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[filter_width]))
        
        z = embedding
        for _ in range(num_layers):
            conv = tf.nn.conv1d(
                value=z,
                filters=W,
                stride=1,
                padding='SAME')
            bias = tf.nn.bias_add(conv, b)
            c = tf.nn.relu(bias)

            d = tf.nn.dropout(c, 0.75)
            #Add BatchNormalization or LocalResponseNormalization
            e = tf.expand_dims(d, 1)

            z = tf.nn.local_response_normalization(
                e,
                depth_radius=5,
                bias=1,
                alpha=0.001,
                beta=0.75
            )
            z = tf.squeeze(z, 1)

        #endfor

        return z

    def _create_maxpooling_layer(self, convolution):
        conv_size      = convolution.shape[1].value
        embedding_size = convolution.shape[2].value

        convolution = tf.expand_dims(convolution, -1)
        pooled = tf.nn.max_pool(
            value=convolution,
            ksize=[1, conv_size, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID')
        
        flat = tf.reshape(pooled, [-1, embedding_size])
        return flat


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
