import tensorflow as tf

from Neon import Neon

layers = tf.contrib.layers
metrics = tf.metrics
arg_scope = tf.contrib.framework.arg_scope


class Magnesium(Neon):

    def __init__(self, neutron, session, graph=tf.Graph(), train_dir='/tmp/neon/'):
        super(Magnesium, self).__init__(neutron, session, graph, train_dir)

    def load_batch(self, batch_size, is_training, num_threads):
        return self.neutron.load_batch(batch_size, is_training, num_threads)

    def model(self, input_data, num_classes, is_training):
        pass

    def losses(self, one_hot_targets, logits):
        pass

    def evaluate(self, checkpoint_dir, checkpoint_name=None, batch_size=32):
        pass
