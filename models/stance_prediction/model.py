import numpy as np
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import Input
from tensorflow.keras.models import Sequential
import tensorflow as tf
from pathlib import Path
from datasets.datasetProcessing.basic.loadDataset import (
    load_label_validation_dataset,
    load_label_training_dataset,
)
from models.utils import get_highest_count, setup_tensorboard

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

_model_dir = (
    Path(os.path.realpath(__file__)).resolve().parents[1]
    / "trained_models/stance_predictor"
)


class TwoLayerStancePredictor(tf.keras.Model):
    def __init__(self, units):
        super(TwoLayerStancePredictor, self).__init__()
        self.layer_1 = Dense(units, activation="relu")
        self.layer_2 = Dense(units, activation="relu")
        self.dropout = Dropout(0.5)
        self.classifier = Dense(1, activation="sigmoid")

    def call(self, inputs, training=None):
        x = self.dropout(inputs, training=training)
        x = self.layer_1(x)
        x = self.dropout(x, training=training)
        x = self.layer_2(x)
        x = self.dropout(x, training=training)
        return self.classifier(x)


def load():
    count = get_highest_count(_model_dir)
    path = str(_model_dir / "TwoLayerStancePredictor_{}".format(count))
    model = tf.keras.models.load_model(path)
    return model


def save(model):
    count = get_highest_count(_model_dir) + 1
    path = str(_model_dir / "TwoLayerStancePredictor_{}".format(count))
    model.save(path)
    print("model saved to {}".format(path))


def train(model, dataset_type, batch_size, epochs, class_weight={0: 1, 1: 1}):
    dataset = (
        load_label_training_dataset(dataset_type)
        .shuffle(10000)
        .batch(batch_size, drop_remainder=True)
    )
    validation_dataset = (
        load_label_validation_dataset(dataset_type)
        .shuffle(10000)
        .batch(batch_size, drop_remainder=True)
    )

    model.fit(
        dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        class_weight=class_weight,
    )

    return model


def initialize_model(batch_size, units, learning_rate):
    loss = tf.keras.losses.BinaryCrossentropy()
    m = TwoLayerStancePredictor(units)
    m.build((batch_size, 1536))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    m.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    m.summary()
    return m


def main():
    BATCH_SIZE = 32
    m = initialize_model(BATCH_SIZE, 256)

    m = train(m, "fever", BATCH_SIZE, 3)
    m = train(m, "scifact", BATCH_SIZE, 10)

    save(m)

    loaded_model = load()
    loaded_model.summary()


if __name__ == "__main__":
    main()
