import tensorflow as tf
import os
import pandas as pd
from csv import reader
from pathlib import Path
import numpy as np


dataset_path = Path(os.path.realpath(__file__)).resolve().parents[1] / "trainingDataset"

def read_tfrecord(serialized_example):
    feature_description = {
        'X': tf.io.FixedLenFeature([], tf.float32),
        'Y': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    X = example['X']
    Y = example['Y']

    return X, Y


def load_dataset():
    filenames = [str(dataset_path / x) for x in os.listdir(dataset_path)]
    dataset = tf.data.TFRecordDataset(filenames)

    parsed_image_dataset = dataset.map(read_tfrecord)

    for e in parsed_image_dataset.take(1):
       print(e)


if __name__ == '__main__':
    load_dataset()
    