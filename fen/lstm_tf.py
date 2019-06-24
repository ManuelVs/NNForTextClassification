import tensorflow as tf


class LSTMModel:
    def __init__(self, embedding):
        self._embedding = embedding

    def __call__(self, input):
        self._embedding_tf = self._create_embedding_layer(
            self._embedding, input)
        self._lstm_tf = self._create_lstm_layer(self._embedding_tf)

        return self._lstm_tf

    def summary(self):
        print('embedding:', str(self._embedding_tf.shape))
        print('lstm:', str(self._lstm_tf.shape))

    def _create_embedding_layer(self, embedding_array, input_x):
        embedding = tf.Variable(
            initial_value=embedding_array)

        embedded_chars = tf.nn.embedding_lookup(
            embedding, tf.cast(input_x, 'int32'))

        return embedded_chars

    def _create_lstm_layer(self, embedding_input):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(100)
        sequence = tf.unstack(embedding_input, axis=1)
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

    model = LSTMModel(embedding)
    model(data)
    model.summary()
