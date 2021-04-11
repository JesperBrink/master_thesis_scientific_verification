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
import argparse
import os
from sentence_transformers import SentenceTransformer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

_model_dir = (
    Path(os.path.realpath(__file__)).resolve().parents[1]
    / "trained_models/stance_predictor"
)


class TwoLayerDenseSrancePredictor:
    def __init__(self, threshold=0.5):
        self.sbert = SentenceTransformer("stsb-distilbert-base")
        self.threshold = threshold
        self.model = load()
        self.model.summary()

    def __call__(self, claim, selected_sentences, selected_abstracts):
        claim_embedding = tf.reshape(tf.constant(self.sbert.encode(claim)), [1, -1])
        res = {}
        for doc_id, rationale_indices in selected_sentences.items():
            rationales = tf.gather(selected_abstracts[doc_id], rationale_indices).numpy().tolist()
            embeddings = tf.constant(list(map(lambda a: self.sbert.encode(str(a)), rationales)))
            claim_column = tf.repeat(claim_embedding, [embeddings.shape[0]], axis=0)
            datapoints = tf.concat([claim_column, embeddings], 1)
            classification = tf.where(
                tf.greater(tf.reduce_mean(self.model(datapoints)), self.threshold),
                "SUPPORT",
                "CONTRADICT",
            ).numpy()
            (classification)

            res[doc_id] = {
                "sentences": rationale_indices,
                "label": classification.decode('ascii')
            }
        print(res)
        return res


def two_layer_stance_predictor(units, dropout=0.5):
    inputs = tf.keras.Input(shape=(1536,), dtype="float64", name="input")
    dropout_first = tf.keras.layers.Dropout(dropout, name="dropout_1")(inputs)
    dense_first = tf.keras.layers.Dense(units, activation="relu", name="dense_1")(
        dropout_first
    )
    dropout_second = tf.keras.layers.Dropout(dropout, name="dropout_2")(dense_first)
    dense_second = tf.keras.layers.Dense(units, activation="relu", name="dense_2")(
        dropout_second
    )
    dropout_third = tf.keras.layers.Dropout(dropout, name="dropout_3")(dense_second)
    outputs = tf.keras.layers.Dense(1, activation="relu", name="output")(dropout_third)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="two_layer_dense_stance_prediction"
    )

    model.summary()

    return model


def load():
    count = get_highest_count(_model_dir)
    path = str(_model_dir / "TwoLayerStancePredictor_{}".format(count))
    model = tf.keras.models.load_model(path)
    print("model loaded from {}".format(path))
    return model


def save(model):
    count = get_highest_count(_model_dir) + 1
    path = str(_model_dir / "TwoLayerStancePredictor_{}".format(count))
    model.save(path)
    print("model saved to {}".format(path))


def train(model, dataset_type, epochs=10, batch_size=32, class_weight={0: 1, 1: 1}):
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


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t", "--train", action="store_true", help="will train the model if set"
    )
    parser.add_argument(
        "-u",
        "--dense_units",
        type=int,
        help="The number of units in the dense layer",
        default=512,
    )
    parser.add_argument(
        "-e", "--epochs", type=int, help="the number of epochs", default=10
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, help="the batch_size", default=32
    )
    parser.add_argument(
        "-w",
        "--work",
        action="store_true",
        help="will run a small test of the evaluator. Can be used to test load and senetence selection",
    )
    args = parser.parse_args()
    if args.train:
        m = two_layer_stance_predictor(args.dense_units)
        loss = tf.keras.losses.BinaryCrossentropy()
        m.compile(optimizer="adam", loss=loss)
        m = train(m, "scifact", batch_size=args.batch_size, epochs=args.epochs)
        save(m)
        m = load()
        m.summary()
    if args.work:
        predictor = TwoLayerDenseSrancePredictor()
        selected_sentences = {1: [2,1]}
        abstract= {1:[
                "redwine is also good",
                "beer is love beer is life",
                "mikkel mikkel and jesper"
            ]
        }
        predictor("gd is not good", selected_sentences, abstract)


if __name__ == "__main__":
    main()
