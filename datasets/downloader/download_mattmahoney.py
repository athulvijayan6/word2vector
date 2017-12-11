import os
import urllib
import collections

import pickle

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, required_bytes=31344016):
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if not os.path.isfile(filename):
        filename, _ = urllib.request.urlretrieve(url + os.path.basename(filename), filename)
    file_info = os.stat(filename)
    if file_info.st_size == required_bytes:
        print("data ready")
    else:
        raise Exception("size not as required")
    return filename


def save_vocabulary(save_name, vocabulary, reversed_vocabulary):
    with open(save_name, 'wb+') as f:
        pickle.dump([vocabulary, reversed_vocabulary], f)


def restore_vocabulary(save_name):
    if not os.path.isfile(save_name):
        raise Exception("saved vocabulary not found")
    with open(save_name, 'rb') as f:
        dictionary, reversed_dictionary = pickle.load(f)
    return dictionary, reversed_dictionary


def create_dataset(words, vocabulary_size):
    count = [["UNK", -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        if word not in dictionary:
            dictionary[word] = len(dictionary)
    # transform data to indices of vocabulary
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

