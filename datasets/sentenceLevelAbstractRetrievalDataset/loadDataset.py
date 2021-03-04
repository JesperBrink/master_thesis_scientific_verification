import tensorflow as tf
import os
import pandas as pd
from csv import reader
from pathlib import Path
import numpy as np


dataset_path = Path(os.path.realpath(__file__)).resolve().parents[1] / "trainingDataset"
validation_path = Path(os.path.realpath(__file__)).resolve().parents[1] / "validationDataset"

def read_tfrecord(serialized_example):
    feature_description = {
        'X': tf.io.FixedLenFeature([1536], tf.float32),
        'Y': tf.io.FixedLenFeature([1], tf.int64),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    X = example['X']
    Y = example['Y']

    return X, Y


def load_dataset():
    filenames = [str(dataset_path / x) for x in os.listdir(dataset_path)]
    dataset = tf.data.TFRecordDataset(filenames)

    return dataset.shuffle(30000).map(read_tfrecord)

def load_validation_dataset():
    filenames = [str(validation_path / x) for x in os.listdir(validation_path)]
    dataset = tf.data.TFRecordDataset(filenames)

    return dataset.shuffle(30000).map(read_tfrecord)


if __name__ == '__main__':
    load_dataset()
    