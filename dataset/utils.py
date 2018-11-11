"""utility functions to generate features and labels"""
import tensorflow as tf

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64list_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_feature_from_array(nparray):
    """
        return a dict with tf.train.Feature as values given a nparray
    """
    return {
        'feature_shape': _int64list_feature(nparray.shape),
        'feature_depth': _bytes_feature(nparray.dtype.name.encode()),
        'feature_data': _bytes_feature(nparray.tobytes())
    }

def get_label_from_array(nparray):
    """
        return a dict with tf.train.Feature as values given a nparray
    """
    return {
        'label_shape': _int64list_feature(nparray.shape),
        'label_depth': _bytes_feature(nparray.dtype.name.encode()),
        'label_data': _bytes_feature(nparray.tobytes())
    }
