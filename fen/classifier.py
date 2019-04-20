import tensorflow as tf


class Classifier:

    def __init__(self, model, input_length, output_length):
        self.model = model
        self.input_length = input_length
        self.output_length = output_length

    def compile(self, batch_size=32):
        self._ds_x = tf.placeholder(tf.float32, [None, self.input_length])
        self._ds_y = tf.placeholder(tf.float32, [None, self.output_length])

        ds = tf.data.Dataset.from_tensor_slices((self._ds_x, self._ds_y))
        ds = ds.batch(batch_size)

        self._ds_it = ds.make_initializable_iterator()
        self._input, self._labels = self._ds_it.get_next()

        self._features = self.model(self._input)
        self._output = _create_dense_layer(self._features, self.output_length)

        self._create_acc_computations()
        self._create_backpropagation()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

    def _create_acc_computations(self):
        self.predictions = tf.argmax(self._output, 1)
        labels = tf.argmax(self._labels, 1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predictions, labels), 'float32'))

    def _create_backpropagation(self):
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self._output,
            labels=self._labels)
        self.loss = tf.reduce_mean(losses)

        optimizer = tf.train.AdamOptimizer(0.001)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        grads_and_vars = optimizer.compute_gradients(self.loss)

        self.train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

    def summary(self):
        print('input:', self._input.shape)
        self.model.summary()
        print('output:', self._output.shape)

    def train(self, X_train, y_train, X_eval, y_eval, epochs=10):
        import time

        for e in range(epochs):
            start_time = time.time()
            loss, acc = self._train(X_train, y_train)
            duration = time.time() - start_time

            val_loss, val_acc = self._eval(X_eval, y_eval)

            output = 'Epoch: {}, loss = {:.4f}, acc = {:.4f}, val_loss = {:.4f}, val_acc = {:.4f}, Time = {:.2f}s'
            print(output.format(e + 1, loss, acc, val_loss, val_acc, duration))
        # endfor

    def _train(self, X_train, y_train):
        import numpy as np

        self.session.run(
            fetches=self._ds_it.initializer,
            feed_dict={
                self._ds_x: X_train,
                self._ds_y: y_train
            })
        loss, acc, = [], []
        while True:
            try:
                _, vloss, vacc = self.session.run(
                    fetches=[self.train_op, self.loss, self.accuracy])

                loss.append(vloss)
                acc.append(vacc)
            except tf.errors.OutOfRangeError:
                break
        # endwhile

        loss, acc = np.mean(loss), np.mean(acc)
        return loss, acc

    def _eval(self, X_val, y_val):
        self.session.run(
            fetches=self._ds_it.initializer,
            feed_dict={
                self._ds_x: X_val,
                self._ds_y: y_val
            })

        loss, acc, = 0, 0
        while True:
            try:
                l, vloss, vacc = self.session.run(
                    fetches=[self._labels, self.loss, self.accuracy])

                loss += vloss * len(l)
                acc += vacc * len(l)
            except tf.errors.OutOfRangeError:
                break

        return loss / len(X_val), acc / len(X_val)

    def predict(self, X):
        import numpy as np

        self.session.run(self._ds_it.initializer,
                         feed_dict={
                             self._ds_x: X,
                             self._ds_y: np.empty((len(X), self.output_length))
                         }
                         )

        pred = list()
        while True:
            try:
                ppred = self.session.run(tf.nn.softmax(self._output))

                pred.extend(map(lambda l: l.tolist(), ppred))
            except tf.errors.OutOfRangeError:
                break

        return pred


def _create_dense_layer(x, output_length):
    '''Creates a dense layer
    '''
    input_size = x.shape[1].value
    W = tf.Variable(
        initial_value=tf.truncated_normal(
            shape=[input_size, output_length],
            stddev=0.1))
    b = tf.Variable(
        initial_value=tf.truncated_normal(
            shape=[output_length]))

    dense = tf.nn.xw_plus_b(x, W, b)

    return dense


if __name__ == '__main__':
    pass
