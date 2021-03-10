import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Input
from tensorflow.keras.models import Sequential
from datasets.datasetProcessing.loadDataset import (
    load_relevance_training_dataset,
    load_relevance_validation_dataset,
)
from models.utils import get_highest_count, setup_tensorboard
import tensorflow as tf
import datetime
import shutil
from pathlib import Path

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

_model_dir = Path(os.path.realpath(__file__)).resolve().parents[1] / "trained_models/abstract_retriever"


class TwoLayerAbstractRetriever(tf.keras.Model):
    def __init__(self, units):
        super(TwoLayerAbstractRetriever, self).__init__()
        self.layer_1 = Dense(units, activation="relu")
        self.layer_2 = Dense(units, activation="relu")
        self.classifier = Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        return self.classifier(x)


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


def train(model, dataset_type, callbacks, BATCH_SIZE):
    dataset = load_relevance_training_dataset(dataset_type).shuffle(10000).batch(
        BATCH_SIZE, drop_remainder=True
    )
    validation_dataset = load_relevance_validation_dataset(dataset_type).shuffle(10000).batch(
        BATCH_SIZE, drop_remainder=True
    )

    model.fit(
        dataset,
        validation_data=validation_dataset,
        epochs=10,
        callbacks=callbacks,
        class_weight={0: 100, 1: 1},
    )

    return model


def main():
    BATCH_SIZE = 32
    tensorboard_callback = setup_tensorboard()
    loss = tf.keras.losses.BinaryCrossentropy()
    m = TwoLayerAbstractRetriever(1024)
    m.build((BATCH_SIZE, 1536))
    m.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    m.summary()

    m = train(m, "fever", [tensorboard_callback], BATCH_SIZE)
    m = train(m, "scifact", [tensorboard_callback], BATCH_SIZE)   

    save(m)

    loaded_model = load()
    loaded_model.summary()


if __name__ == "__main__":
    main()
