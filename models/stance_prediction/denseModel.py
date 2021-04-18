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
import jsonlines
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

_model_dir = (
    Path(os.path.realpath(__file__)).resolve().parents[1]
    / "trained_models/stance_predictor/dense"
)



class TwoLayerDenseSrancePredictor:
    def __init__(self, corpus_path, claim_path, threshold=0.5):
        self.threshold = threshold
        self.model = load()
        self.model.summary()
        self.doc_id_to_abst_embedding_map = self.create_id_to_abstract_map(corpus_path)
        self.id_to_claim_embedding_map = self.create_id_to_claim_map(claim_path)

    def __call__(self, claim_object, selected_sentences, selected_abstracts):
        claim_id = claim_object["id"]
        claim_embedding = tf.reshape(tf.constant(self.id_to_claim_embedding_map[claim_id]), [1,-1])
        res = {}
        for doc_id, rationale_indices in selected_sentences.items():
            rationale_embeddings = tf.gather(self.doc_id_to_abst_embedding_map[doc_id], rationale_indices)
            claim_column = tf.repeat(claim_embedding, [rationale_embeddings.shape[0]], axis=0)
            datapoints = tf.concat([claim_column, rationale_embeddings], 1)
            classification = tf.where(
                tf.greater(tf.reduce_mean(self.model(datapoints)), self.threshold),
                "SUPPORT",
                "CONTRADICT",
            ).numpy()

            res[doc_id] = {
                "sentences": rationale_indices,
                "label": classification.decode('ascii')
            }
        return {"id": claim_id, "evidence": res}

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


def two_layer_stance_predictor(units, dropout=0.5):
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
    parser.add_argument(
        "-cl",
        "--claim_embedding",
        type=str
    )
    parser.add_argument(
        "-co",
        "--corpus_embedding",
        type=str
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
        predictor = TwoLayerDenseSrancePredictor(args.corpus_embedding, args.claim_embedding)
        selected_sentences = {4983: [2,1]}
        abstract= {4983:[
                "redwine is also good",
                "beer is love beer is life",
                "mikkel mikkel and jesper"
            ]
        }
        print(predictor({"id": 1, "claim": "this is not used"}, selected_sentences, abstract))


if __name__ == "__main__":
    main()
