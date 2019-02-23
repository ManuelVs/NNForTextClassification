
import tensorflow as tf

filenames = ["data/cifar/cifar-10-python.tar.gz"]

dataset = tf.data.TFRecordDataset(
    filenames=filenames,
    compression_type="GZIP")

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as session:
    session.run(iterator.initializer)

    x_batch, y_batch = session.run(next_element)

    print(x_batch)
