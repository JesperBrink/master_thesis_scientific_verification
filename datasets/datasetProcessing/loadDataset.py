import tensorflow as tf
import os
import pandas as pd
from csv import reader
from pathlib import Path
import numpy as np


trainint_set_path = (
    Path(os.path.realpath(__file__)).resolve().parents[1] / "trainingDataset"
)
validation_set_path = (
    Path(os.path.realpath(__file__)).resolve().parents[1] / "validationDataset"
)


def read_relevance_tfrecord(serialized_example):
    feature_description = {
        "X": tf.io.FixedLenFeature([1536], tf.float32),
        "relevance": tf.io.FixedLenFeature([1], tf.int64),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    X = example["X"]
    Y = example["relevance"]

    return X, Y


def load_relevance_training_dataset(dataset_type):
    filenames = [str(trainint_set_path / x) for x in os.listdir(trainint_set_path) if dataset_type in x]
    dataset = tf.data.TFRecordDataset(filenames)

    return dataset.shuffle(30000).map(read_relevance_tfrecord)


def load_relevance_validation_dataset(dataset_type):
    filenames = [str(validation_set_path / x) for x in os.listdir(validation_set_path) if dataset_type in x]
    dataset = tf.data.TFRecordDataset(filenames)

    return dataset.shuffle(30000).map(read_relevance_tfrecord)


def read_label_tfRecord(serialized_example):
    feature_description = {
        "X": tf.io.FixedLenFeature([1536], tf.float32),
        "relevance": tf.io.FixedLenFeature([1], tf.int64),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    X = example["X"]
    Y = example["label"]

    return X, Y


def load_label_training_dataset(dataset_type):
    filenames = [str(trainint_set_path / x) for x in os.listdir(trainint_set_path) if "not_relevant" not in x and dataset_type in x]
    dataset = tf.data.TFRecordDataset(filenames)

    return dataset.shuffle(150000).map(read_label_tfRecord)


def load_label_validation_dataset(dataset_type):
    filenames = [str(validation_set_path / x) for x in os.listdir(validation_set_path) if "not_relevant" not in x and dataset_type in x]
    dataset = tf.data.TFRecordDataset(filenames)

    return dataset.shuffle(150000).map(read_label_tfRecord)
