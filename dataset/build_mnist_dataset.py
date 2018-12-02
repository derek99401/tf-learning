"""Read MNIST Feature file"""
import struct
import argparse
import tfrecords_dataset_builder as builder
import numpy as np
from utils import get_feature_from_array, get_label_from_array

def _read_int32(fr):
    chunk = fr.read(4)
    return struct.unpack('>i', chunk)[0]

def read_mnist_image_file(filepath):
    """
        read MNIST image file as np array,
        return N x rows x cols np array
    """
    with open(filepath, 'rb') as fr:
        #read magic number
        magic_number = _read_int32(fr)
        assert magic_number == 2051

        n_images = _read_int32(fr)
        rows = _read_int32(fr)
        cols = _read_int32(fr)
        img_size = rows*cols
        print('open ', filepath, 'n_images', n_images, 'rows', rows, 'cols', cols)

        data = fr.read()
        assert len(data) == img_size*n_images
        images = np.frombuffer(data, dtype=np.uint8).reshape((n_images, rows, cols))
        return images

def read_mnist_label_file(filepath):
    """
        read MNIST label file as np array,
        return np array of size N,
        where N is the number of labels
    """
    with open(filepath, 'rb') as fr:
        #read magic number
        magic_number = _read_int32(fr)
        assert magic_number == 2049

        n_items = _read_int32(fr)
        print('open ', filepath, 'n_items', n_items)

        data= fr.read()
        assert n_items == len(data)
        return np.frombuffer(data, dtype=np.uint8)

class MnistImages(object):
    def __init__(self, mnist_image_filepath):
        self.images = read_mnist_image_file(mnist_image_filepath)

    def __iter__(self):
        for i in range(self.images.shape[0]):
            yield get_feature_from_array(self.images[i, :, :])

    def size(self):
        return self.images.shape[0]

class MnistLabels(object):
    def __init__(self, mnist_label_filepath):
        self.labels = read_mnist_label_file(mnist_label_filepath)

    def __iter__(self):
        for i in range(self.labels.size):
            label_vector = np.zeros(10, dtype=np.uint8)
            label_vector[self.labels[i]] = 1
            yield get_label_from_array(label_vector)

    def size(self):
        return self.labels.size

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--images', required=True, help='MNIST image file')
    arg_parser.add_argument('--labels', required=True, help='MNIST label file')
    arg_parser.add_argument('-o', '--output', required=True, help='tfrecords output')
    args = arg_parser.parse_args()
    feature_gen = MnistImages(args.images)
    label_gen = MnistLabels(args.labels)
    assert feature_gen.size() > 0
    assert label_gen.size() > 0
    assert feature_gen.size() == label_gen.size()
    builder.build_with_feature_and_label(args.output, feature_gen, label_gen)
    print('build tfrecords to ', args.output)
