import collections
import random
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_skipgrams_to_tfrecord(data, writer, num_skips=1, skip_window=2):
    """
    Writes a list of tokens to a TFRecordWriter
    :param data: list of tokens
    :param writer: TFRecordWriter to write into
    :param num_skips: How many examples to extract from a target
    :param skip_window: skipgram window
    :return: null
    """
    assert num_skips <= 2 * skip_window
    data_index = 0
    span = 2 * skip_window + 1
    print("creating skipgram tfrecords")
    while data_index + span < len(data):
        buffer = collections.deque(maxlen=span)
        buffer.extend(data[data_index: data_index + span])
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            # input is the centre of buffer
            input_data = buffer[skip_window]
            # output is the target we trying to predict
            target_data = buffer[target]
            # create tfrecord example
            feature = {
                'input': _int64_feature(input_data),
                'target': _int64_feature(target_data)
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        data_index += 1
    print("done writing skipgram tfrecords file")


