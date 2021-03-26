import os
from pathlib import Path

import tensorflow as tf
from models.utils import get_highest_count, setup_tensorboard
from datasets.datasetProcessing.lstm.loadDataset import read_lstm_tfrecord_dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

_model_dir = (
    Path(os.path.realpath(__file__)).resolve().parents[1]
    / "trained_models/sentence_selection"
)


class TwoLayerAbstractRetriever(tf.keras.Model):
    def __init__(self, units):
        super(TwoLayerAbstractRetriever, self).__init__()
        self.layer_1 = tf.keras.layers.LSTM(
            units, recurrent_activation="relu", recurrent_initializer="glorot_uniform"
        )
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.layer_1(inputs)
        return self.classifier(x)


def train(model, batch_size, epochs):
    dataset = (
        read_lstm_tfrecord_dataset()
        .shuffle(10000)
        .batch(batch_size, drop_remainder=True)
    )

    model.fit(
        dataset,
        epochs=epochs,
    )

    return model


def load():
    count = get_highest_count(_model_dir)
    path = str(_model_dir / "TwoLayerAbstractRetriever_{}".format(count))
    model = tf.keras.models.load_model(path)
    return model

def save(model):
    count = get_highest_count(_model_dir) + 1
    path = str(_model_dir / "TwoLayerAbstractRetriever_{}".format(count))
    model.save(path)
    print("model saved to {}".format(path))


def initialize_model(batch_size, units):
    loss = tf.keras.losses.BinaryCrossentropy()
    m = TwoLayerAbstractRetriever(units)
    m.build((batch_size, None, 1536))
    m.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    m.summary()
    return m


if __name__ == "__main__":
    BATCH_SIZE = 1

    m = initialize_model(BATCH_SIZE, 512)

    m = train(m, BATCH_SIZE, 1)

    save(m)

    loaded_model = load()
    loaded_model.summary()
