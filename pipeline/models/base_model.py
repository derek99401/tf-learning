from .model_interface import ModelInterface
import tensorflow as tf

class BaseModel(ModelInterface):

    def __init__(self):
        pass

    def _map_impl(self, serialized):
        features = tf.io.parse_single_example(
            serialized,
            {
                'feature_shape': tf.VarLenFeature(tf.int64),
                'feature_data': tf.FixedLenFeature([], tf.string),
                'label_shape': tf.VarLenFeature(tf.int64),
                'label_data': tf.FixedLenFeature([], tf.string),
            },
            'parse_example'
        )
        feature = tf.decode_raw(features['feature_data'], tf.uint8)
        label = tf.decode_raw(features['label_data'], tf.uint8)
        feature_shape = tf.sparse.to_dense(features['feature_shape'])
        label_shape = tf.sparse.to_dense(features['label_shape'])
        feature = tf.reshape(feature, feature_shape)
        label = tf.reshape(label, label_shape)
        return [feature, label]


    def map_input(self, dataset, num_parallel_calls):
        mapped_dataset = dataset.map(self._map_impl, num_parallel_calls=num_parallel_calls)
        return mapped_dataset

