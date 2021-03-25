import os 
from pathlib import Path

import tensorflow as tf

trainint_set_path = (
    Path(os.path.realpath(__file__)).resolve().parents[1] / "LstmTrainingDataset"
)
validation_set_path = (
    Path(os.path.realpath(__file__)).resolve().parents[1] / "LstmValidationDataset"
)

def read_relevance_tfrecord(serialized_example):
    feature_description = {
        "sequence": tf.io.VarLenFeature(tf.float32),
        "label": tf.io.VarLenFeature(tf.int64),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    X = example["sequence"]
    Y = example["label"]

    return X, Y

def read_lstm_tfrecord_dataset():
    filenames = [
        str(trainint_set_path / x)
        for x in os.listdir(trainint_set_path)
    ]
    dataset = tf.data.TFRecordDataset(filenames)

    return dataset.shuffle(30000).map(read_relevance_tfrecord)

if __name__ == '__main__':
    count = 0
    for x in read_lstm_tfrecord_dataset():
        assert x[0].shape[0] / x[1].shape[0] == 768*2
        count+=1
    print(count)

    