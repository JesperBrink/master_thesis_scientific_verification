import numpy as np
from datasets.datasetProcessing.basic.loadDataset import (
    load_relevance_training_dataset,
    load_relevance_validation_dataset,
)
from models.utils import get_highest_count, setup_tensorboard
import tensorflow as tf
from pathlib import Path
import jsonlines
import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

_model_dir = (
    Path(os.path.realpath(__file__)).resolve().parents[1]
    / "trained_models/sentence_selection/dense"
)


class TwoLayerDenseSentenceSelector:
    def __init__(self, corpus_embedding_path, claim_embedding_path, threshold=0.5, model=None):
        self.threshold = threshold
        self.doc_id_to_abst_embedding_map = self.create_id_to_abstract_map(corpus_embedding_path)
        self.id_to_claim_embedding_map = self.create_id_to_claim_map(claim_embedding_path)
        if model is None:
            self.model = load()
        else:
            self.model = model
        self.model.summary()

    def __call__(self, claim, abstracts):
        result = {}

        claim_embedding = tf.constant(self.id_to_claim_embedding_map[claim["id"]])
        for doc_id, _ in abstracts.items():
            abstract_embedding = tf.constant(self.doc_id_to_abst_embedding_map[doc_id])
            stacked_claim = tf.repeat(
                [claim_embedding], abstract_embedding.shape[0], axis=0
            )
            datapoints = tf.concat([stacked_claim, abstract_embedding], 1)
            model_result = tf.reshape(self.model(datapoints), (-1))
            top_k, indices = tf.math.top_k(model_result, k=3)
            res = tf.reshape(
                tf.gather(indices, tf.where(top_k > self.threshold), axis=0), (-1)
            )
            rationales = res.numpy().tolist()
            if len(rationales) < 1:
                continue
            result[doc_id] = rationales

        return result

    def create_id_to_abstract_map(self, corpus_path):
        abstract_id_to_abstract = dict()
        corpus = jsonlines.open(corpus_path)
        for data in corpus:
            abstract_id_to_abstract[data["doc_id"]] = data["abstract"]

        return abstract_id_to_abstract

    def create_id_to_claim_map(self, claim_path):
        claim_id_to_embeding = dict()
        claims = jsonlines.open(claim_path)
        for data in claims:
            claim_id_to_embeding[data["id"]] = data["claim"]

        return claim_id_to_embeding


def check_for_folder():
    if not _model_dir.exists():
        print("Creating save folder")
        os.makedirs(_model_dir)


def two_layer_sentence_selector(units, dropout=0.5):
    check_for_folder()
    
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
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(dropout_third)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="two_layer_dense_sentence_selection"
    )

    model.summary()

    return model


def load():
    count = get_highest_count(_model_dir)
    path = str(_model_dir / "TwoLayerAbstractRetriever_{}".format(count))
    model = tf.keras.models.load_model(path)
    print("model loaded from {}".format(path))
    return model


def save(model):
    count = get_highest_count(_model_dir) + 1
    path = str(_model_dir / "TwoLayerAbstractRetriever_{}".format(count))
    model.save(path)
    print("model saved to {}".format(path))


def train(model, dataset_type, batch_size, epochs, class_weight={0: 1, 1: 1}):
    dataset = (
        load_relevance_training_dataset(dataset_type)
        .shuffle(10000)
        .batch(batch_size, drop_remainder=True)
    )
    validation_dataset = (
        load_relevance_validation_dataset(dataset_type)
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


def setup_for_training(dense_units, learning_rate):
    m = two_layer_sentence_selector(dense_units)
    loss = tf.keras.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    m.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return m


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

    parser.add_argument("-cl", "--claim_embedding", type=str)
    parser.add_argument("-co", "--corpus_embedding", type=str)
    args = parser.parse_args()
    if args.train:
        m = setup_for_training(args.dense_units, 0.001)
        m = train(m, "scifact", batch_size=args.batch_size, epochs=args.epochs)
        save(m)
        m = load()
        m.summary()
    if args.work:
        selector = TwoLayerDenseSentenceSelector(
            args.corpus_embedding, args.claim_embedding, threshold=0.3
        )
        abstracts = {4983: [""]}
        print(selector({"id": 13, "claim": "gd is not"}, abstracts))


if __name__ == "__main__":
    main()
