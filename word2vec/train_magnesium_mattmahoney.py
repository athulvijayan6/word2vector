import os
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from neutrons.neutron_mattmahoney import NeutronMattMahoney

AI_HOME = os.environ['AI_HOME']
AI_DATA = os.environ['AI_DATA']

data_dir = os.path.join(AI_DATA, 'mattmahoney')
train_dir = os.path.join(AI_DATA, 'word2vec', 'model')


def test_input():
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    graph = tf.Graph()
    neutron = NeutronMattMahoney(data_dir, graph)
    neutron.download_and_convert_skipgram()
    reversed_dictionary = neutron.load_reversed_vocabulary()
    with tf.Session(graph=graph) as session:
        with graph.as_default():
            # Start queues to fetch data
            batch_size = 32
            inputs, targets = neutron.load_batch(batch_size=batch_size, is_training=True)
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=session, coord=coord)
            session.run(tf.global_variables_initializer())
            try:
                inputs, targets = session.run([inputs, targets])
                print(inputs)
                print(reversed_dictionary[inputs[0]])
                print(reversed_dictionary[targets[0]])
            except OutOfRangeError as e:
                print("out of range")


if __name__ == "__main__":
    test_input()