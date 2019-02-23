import tensorflow as tf

class Classifier:

    def __init__(self, model):
        self.model        = model
        num_classes       = model.output.shape[1].value
        self.input_labels = tf.placeholder(tf.float32, [None, num_classes])

        self._create_acc_computations()
        self._create_backpropagation()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())


    def _create_acc_computations(self):
        self.predictions  = tf.argmax(self.model.output, 1)
        labels            = tf.argmax(self.input_labels, 1)
        self.accuracy     = tf.reduce_mean(tf.cast(tf.equal(self.predictions, labels), 'float32'))
    

    def _create_backpropagation(self):
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.model.output,
            labels=self.input_labels)
        self.loss = tf.reduce_mean(losses)

        optimizer = tf.train.AdamOptimizer(0.001)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        grads_and_vars = optimizer.compute_gradients(self.loss)

        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    def train(self, X_train, y_train, X_eval, y_eval, epochs=10, batch_size=32):
        import time

        for e in range(epochs):
            start_time = time.time()
            loss, acc = self._train(X_train, y_train, batch_size)
            duration = time.time() - start_time

            val_loss, val_acc = self._eval(X_eval, y_eval)

            output = 'Epoch: {}, loss = {:.4f}, acc = {:.4f}, val_loss = {:.4f}, val_acc = {:.4f}, Time = {:.2f}s' 
            print(output.format(e + 1, loss, acc, val_loss, val_acc, duration))
        #endfor

    def _train(self, X_train, y_train, batch_size):
        import numpy as np
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        self.session.run(iterator.initializer)
        loss, acc, = [], []
        while True:
            try:
                x_batch, y_batch = self.session.run(next_element)
                
                _, vloss, vacc = self.session.run(
                    fetches=[self.train_op, self.loss, self.accuracy],
                    feed_dict={
                        self.model.input : x_batch,
                        self.input_labels: y_batch
                    })

                loss.append(vloss)
                acc.append(vacc)
            except tf.errors.OutOfRangeError:
                break
        #endwhile

        loss, acc = np.mean(loss), np.mean(acc)
        return loss, acc

    def _eval(self, x_batch, y_batch):
        loss, acc = self.session.run(
            fetches=[self.loss, self.accuracy],
            feed_dict={
                self.model.input  : x_batch,
                self.input_labels : y_batch,
            }
        )

        return loss, acc

    def predict(self, X):
        y_hat = self._predict(
            x_batch=X
        )

        return y_hat

    def _predict(self, x_batch):
        feed_dict = {
            self.model.input : x_batch,
        }

        y_hat = self.session.run(
            fetches=[self.predictions],
            feed_dict=feed_dict
        )

        return y_hat

if __name__ == '__main__':
    pass
