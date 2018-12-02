"""generate a random dataset"""
import random
import argparse
import tfrecords_dataset_builder as builder
import numpy as np
from utils import get_feature_from_array, get_label_from_array

def feature_gen():
    i = 0
    while i < 100:
        rand_mat = np.random.rand(10,5)
        yield get_feature_from_array(rand_mat)
        i += 1

def label_gen():
    i = 0
    while i < 100:
        rand_label = np.array([random.randint(0, 10)])
        yield get_label_from_array(rand_label)
        i += 1

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("output_dataset")
    args = arg_parser.parse_args()
    builder.build_with_feature_and_label(args.output_dataset, feature_gen(), label_gen())
