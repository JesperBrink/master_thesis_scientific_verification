import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Input
from tensorflow.keras.models import Sequential
from datasets.sentenceLevelAbstractRetrievalDataset.loadDataset import load_dataset, load_validation_dataset
import tensorflow as tf
import datetime
import shutil
from pathlib import Path

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

_model_dir = Path(os.path.realpath(__file__)).resolve().parents[1] / "trained_models"


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
    path = str(_model_dir / 'TwoLayerAbstractRetriever_{}'.format(count))
    model = tf.keras.models.load_model(path)
    return model


def save(model):
    count = get_highest_count(_model_dir) + 1
    path = str(_model_dir / 'TwoLayerAbstractRetriever_{}'.format(count))
    model.save(path)
    print('model saved to {}'.format(path))


def get_highest_count(dir):
    m = -1
    for file in os.listdir(dir):
        number = int(file.split('_')[-1])
        if number > m: 
            m = number
    return m


def setup_tensorboard():
    # clear old logs
    if os.path.exists("./logs"):
        shutil.rmtree("./logs")

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def main():
    BATCH_SIZE = 32
    tensorboard_callback = setup_tensorboard()
    loss = tf.keras.losses.BinaryCrossentropy()
    m = TwoLayerAbstractRetriever(1024)
    m.build((BATCH_SIZE, 1536))
    m.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    m.summary()
    dataset = load_dataset().batch(BATCH_SIZE, drop_remainder=True)
    validation_dataset = load_validation_dataset().batch(BATCH_SIZE, drop_remainder=True)
    # early_stopping =tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    m.fit(
        dataset, 
        validation_data=validation_dataset,
        epochs=17,
        callbacks=[tensorboard_callback],
        class_weight={0:1,1:50}
    )
    save(m)

    loaded_model = load()
    loaded_model.summary()


if __name__ == "__main__":
    main()
