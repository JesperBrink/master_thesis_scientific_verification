import os
from pathlib import Path

import tensorflow as tf

trainint_set_path = (
    Path(os.path.realpath(__file__)).resolve().parents[1] / "LstmTrainingDataset"
)
validation_set_path = (
    Path(os.path.realpath(__file__)).resolve().parents[1] / "LstmValidationDataset"
)


def _read_relevance_tfrecord(serialized_example):
    feature_description = {
        "sequence": tf.io.VarLenFeature(tf.float32),
        "label": tf.io.VarLenFeature(tf.int64),
        "shape": tf.io.FixedLenFeature([2], tf.int64)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)
    X = tf.reshape(tf.sparse.to_dense(example["sequence"]), example["shape"])
    Y = tf.sparse.to_dense(example["label"])

    return X, Y


def read_lstm_tfrecord_dataset(dataset_type):
    paht = trainint_set_path if dataset_type == "train" else validation_set_path
    filenames = [str(trainint_set_path / x) for x in os.listdir(trainint_set_path)]
    dataset = tf.data.TFRecordDataset(filenames)

    return dataset.map(_read_relevance_tfrecord)


def load_lstm_validation_set():
    return read_lstm_tfrecord_dataset("validation")

def load_lstm_train_set():
    return read_lstm_tfrecord_dataset("train")


if __name__ == "__main__":
    count = 0
    for x in load_lstm_train_set():
        assert not tf.reduce_any(tf.math.is_nan(x[0]))
        assert tf.debugging.is_numeric_tensor(x[1])
        assert x[0].shape[1] == 768 * 2