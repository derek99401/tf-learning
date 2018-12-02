"""build tfrecords dataaset from a list of images and labels"""

import tensorflow as tf
from tensorflow.python_io import TFRecordWriter

def build_with_feature(output_file, feature_generator):
    """
        build tfrecords dataset, must provide both feature generator and label generator
        output_file: where output tfrecords should be
        feature_generator: a generator function, should yield a dict,
            where key is string and value is tf.train.Feature
    """
    cnt = 0
    with TFRecordWriter(output_file) as writer:
        for feature in feature_generator:
            sample = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(sample.SerializeToString())
            cnt += 1

def build_with_feature_and_label(output_file, feature_generator, label_generator):
    """
        build tfrecords dataset, must provide both feature generator and label generator
        output_file: where output tfrecords should be
        feature_generator: a generator function, should yield a dict,
            where key is string and value is tf.train.Feature
        label_generator: a generator function, should yield a dict,
            where key is string and value is tf.train.Feature
    """
    cnt = 0
    with TFRecordWriter(output_file) as writer:
        for feature, label in zip(feature_generator, label_generator):
            feature_label = {}
            feature_label.update(feature)
            feature_label.update(label)
            sample = tf.train.Example(features=tf.train.Features(feature=feature_label))
            writer.write(sample.SerializeToString())
            cnt += 1
