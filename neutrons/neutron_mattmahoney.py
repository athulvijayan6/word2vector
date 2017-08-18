import os
import zipfile
import tensorflow as tf

from neutrons.downloader.download_mattmahoney import maybe_download, save_vocabulary, create_dataset, restore_vocabulary
from neutrons.preprocessing.skipgram_preprocess import write_skipgrams_to_tfrecord


class NeutronMattMahoney(object):
    TRAIN_FILE = 'train.tfrecords'
    VALIDATION_FILE = 'validate.tfrecords'
    TEST_FILE = 'test.tfrecords'
    ITEMS_TO_DESCRIPTIONS = {
        'input': 'Integer index to vocabulary.input to word2vec',
        'target': 'Integer index to vocabulary.target for word2vec',
    }

    def __init__(self, data_dir, graph, vocabulary_size=50000):
        self.data_dir = data_dir
        self.graph = graph
        self.vocabulary_size = vocabulary_size
        self.vocabulary_file = os.path.join(self.data_dir, "vocabulary.json")

    def download(self):
        text_data = 'text8.zip'
        required_bytes = 31344016
        filename = os.path.join(self.data_dir, text_data)
        return maybe_download(filename, required_bytes)

    def download_and_convert_skipgram(self):
        vocabulary_size = 50000
        filename = self.download()
        with zipfile.ZipFile(filename) as f:
            vocabulary = tf.compat.as_str(f.read(f.namelist()[0])).split()
        print("Length of vocabulary " + str(len(vocabulary)))
        data, count, dictionary, reversed_dictionary = create_dataset(vocabulary, vocabulary_size)
        save_vocabulary(self.vocabulary_file, dictionary, reversed_dictionary)
        del vocabulary

        # create tfrecord saver for training
        train_filename = os.path.join(self.data_dir, 'train.tfrecords')
        writer = tf.python_io.TFRecordWriter(train_filename)
        write_skipgrams_to_tfrecord(data, writer)
        writer.close()

    def load_vocabulary(self):
        if not os.path.isfile(self.vocabulary_file):
            raise Exception("Dictionary not found")
        else:
            dictionary, _ = restore_vocabulary(self.vocabulary_file)
            return dictionary

    def load_reversed_vocabulary(self):
        if not os.path.isfile(self.vocabulary_file):
            raise Exception("Dictionary not found")
        else:
            _, reversed_dictionary = restore_vocabulary(self.vocabulary_file)
            return reversed_dictionary

    def _preprocess(self, input_data):
        """
        Performs preprocessing to single training input
        :param input_data: input data
        :return: processed output
        """
        return input_data

    def _read_example(self, filename_queue, is_training=True):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'input': tf.FixedLenFeature([], tf.int64),
                                               'target': tf.FixedLenFeature([], tf.int64)
                                           })
        input_data = tf.cast(features['input'], tf.int64)
        input_data = self._preprocess(input_data)
        output_data = tf.cast(features['target'], tf.int64)
        return input_data, output_data

    def load_batch(self, batch_size, is_training=True, num_threads=1):
        with self.graph.as_default():
            filename = self.TRAIN_FILE if is_training else self.VALIDATION_FILE
            filenames = [os.path.join(self.data_dir, filename)]
            with tf.name_scope('neutron'):
                filename_queue = tf.train.string_input_producer(filenames)
                input_data, target_data = self._read_example(filename_queue, is_training=is_training)
                inputs, targets = tf.train.shuffle_batch([input_data, target_data],
                                                         batch_size=batch_size,
                                                         num_threads=num_threads,
                                                         capacity=1000 + 3 * batch_size,
                                                         min_after_dequeue=1000)
                return inputs, targets
