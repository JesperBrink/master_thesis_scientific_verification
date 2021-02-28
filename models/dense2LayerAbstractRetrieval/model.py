import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Input
from tensorflow.keras.models import Sequential
from datasets.sentenceLevelAbstractRetrievalDataset.loadDataset import load_dataset
import tensorflow as tf

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


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
    pass


def save():
    pass


def main():
    BATCH_SIZE = 32
    m = TwoLayerAbstractRetriever(128)
    m.build((BATCH_SIZE, 1536))
    loss = tf.keras.losses.BinaryCrossentropy()
    m.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    m.summary()
    dataset = load_dataset().batch(BATCH_SIZE)
    m.fit(dataset)
    print("wubwubwub: ", m(np.matrix([1])))


if __name__ == "__main__":
    main()
