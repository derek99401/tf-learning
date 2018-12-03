import tensorflow as tf
from enum import Enum

class ActionType(Enum):
    train=1
    test=2
    debug_print=3

class TfOption:
    def __init__(self):
        pass

    @property
    def num_parallel_calls(self):
        return 4

    @property
    def num_parallel_batches(self):
        return 1

    @property
    def prefetch_buffer_size(self):
        return 1

class TfPipeline(object):
    def __init__(self, model):
        self.tf_option_ = TfOption()
        self.model_ = model
        self.dataset = None

    def initialize_input(self):
        with tf.name_scope('input_dataset'):
            filenames = tf.placeholder(tf.string, shape=[None], name='input_filenames')
            batch_size = tf.placeholder(tf.int64, shape=None, name='batch_size')
            files = tf.data.Dataset.list_files(filenames)
            dataset = files.interleave(tf.data.TFRecordDataset, 1)
            dataset = dataset.shuffle(batch_size * 2 + 200)
            dataset = dataset.repeat()
            dataset = self.model_.map_input(dataset, num_parallel_calls=self.tf_option_.num_parallel_calls)
            dataset = dataset.batch(batch_size=batch_size)

            #dataset = dataset.apply(
            #    tf.contrib.data.map_and_batch(
            #        map_func=self.model_.dataset_map_func(),
            #        batch_size=batch_size,
            #        num_parallel_batches=self.tf_option_.num_parallel_batches,
            #        num_parallel_calls=self.tf_option_.num_parallel_calls
            #    )
            #)

            dataset = dataset.prefetch(buffer_size=self.tf_option_.prefetch_buffer_size) #alway have n batch ready to load
            return dataset, filenames, batch_size

    def print_sample(self, sample):
        return tf.print(sample, name='sample', summarize=-1)

    def run_testing(self, sess):
        pass

    def run_training(self, sess):
        pass

    def run_debug_print(self, sess):
        sess.run(self.print_opt)
        sess.run(self.print_opt)

    def run(self, action_type, input_files, train_batch_size=50, max_iter=5e4):
        dataset, filenames, batch_size = self.initialize_input()
        iterator = dataset.make_initializable_iterator()
        self.next_element = iterator.get_next()
        self.print_opt = self.print_sample(self.next_element)
        with tf.Session() as sess:
            sess.run(iterator.initializer, feed_dict={filenames: input_files, batch_size: train_batch_size})
            if action_type == ActionType.train:
                self.run_training(sess)
            elif action_type == ActionType.test:
                self.run_testing(sess)
            elif action_type == ActionType.debug_print:
                self.run_debug_print(sess)
