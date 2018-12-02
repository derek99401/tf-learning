"""build lmdb dataaset from a list of images and labels"""
import tensorflow as tf
import lmdb

class LmdbDatasetBuilder(object):

    def __init__(self):
        pass
        self.map_size = 1*1024*1024*1024 #1GB

    def build_with_feature(self, output_file, feature_generator):
        """
            build lmdb dataset, must provide both feature generator and label generator
            output_file: where output lmdb should be
            feature_generator: a generator function, should yield a dict,
                where key is string and value is tf.train.Feature
        """
        lmdb_env = lmdb.open(output_file, self.map_size)
        cnt = 0
        with lmdb_env.begin(write=True) as txn:
            for feature in feature_generator:
                sample = tf.train.Example(features=tf.train.Features(feature=feature))
                txn.put(str(cnt).encode('ascii'), sample.SerializeToString())
                cnt += 1
        lmdb_env.close()

    def build_with_feature_and_label(self, output_file, feature_generator, label_generator):
        """
            build lmdb dataset, must provide both feature generator and label generator
            output_file: where output lmdb should be
            feature_generator: a generator function, should yield a dict,
                where key is string and value is tf.train.Feature
            label_generator: a generator function, should yield a dict,
                where key is string and value is tf.train.Feature
        """
        lmdb_env = lmdb.open(output_file, self.map_size)
        cnt = 0
        with lmdb_env.begin(write=True) as txn:
            for feature, label in zip(feature_generator, label_generator):
                feature_label = {}
                feature_label.update(feature)
                feature_label.update(label)
                sample = tf.train.Example(features=tf.train.Features(feature=feature_label))
                txn.put(str(cnt).encode('ascii'), sample.SerializeToString())
                cnt += 1

        lmdb_env.close()
