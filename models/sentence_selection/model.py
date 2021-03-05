import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Input
from tensorflow.keras.models import Sequential
from datasets.sentenceLevelAbstractRetrievalDataset.loadDataset import (
    load_relevance_training_dataset,
    load_relevance_validation_dataset,
)
from datasets.sentenceLevelAbstractRetrievalDataset.loadDataset import load_dataset, load_validation_dataset
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
        self.layer_3 = Dense(units, activation="relu")
        self.classifier = Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
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


def main():
    BATCH_SIZE = 32
    tensorboard_callback = setup_tensorboard()
    loss = tf.keras.losses.BinaryCrossentropy()
    m = TwoLayerAbstractRetriever(1024)
    m.build((BATCH_SIZE, 1536))
    m.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    m.summary()
    dataset = load_relevance_training_dataset().batch(BATCH_SIZE, drop_remainder=True)
    validation_dataset = load_relevance_validation_dataset().batch(
        BATCH_SIZE, drop_remainder=True
    )

    m.fit(
        dataset,
        validation_data=validation_dataset,
        epochs=17,
        callbacks=[tensorboard_callback],
        class_weight={0: 1, 1: 50},
    )
    save(m)

    loaded_model = load()
    loaded_model.summary()


if __name__ == "__main__":
    main()
