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
from sentence_transformers import CrossEncoder
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

_model_dir = (
    Path(os.path.realpath(__file__)).resolve().parents[1]
    / "trained_models/sentence_selection/dense"
)


class TwoLayerDenseSentenceSelector:
    def __init__(
            self,
            corpus_embedding_path,
            claim_embedding_path,
            sbert_threshold=0.5,
            k=None,
            model=None,
            cross_encoder_threshold=0.5,
            cross_encoder_path=None,
            corpus_path=None
        ):
        self.sbert_threshold = sbert_threshold
        self.cross_encoder_threshold = cross_encoder_threshold
        self.k = k
        self.id_to_claim_embedding_map = self.create_id_to_claim_map(claim_embedding_path)
        self.id_to_abstract_embedding_map = self.create_id_to_abstract_map(
            corpus_embedding_path
        )

        (
            self.sentence_embeddings,
            self.rationale_id_to_abstract_and_sentence_id_pair,
        ) = self.get_sentence_embeddings_for_all_abstracts(corpus_embedding_path)
        self.number_of_abstracts_in_corpus = self.get_number_of_abstracts_in_corpus(
            corpus_embedding_path
        )    

        if cross_encoder_path:
            self.use_cross_encoder = True
            self.id_to_abstract_map = self.create_id_to_abstract_map(
                corpus_path
            )
            self.cross_encoder = CrossEncoder(cross_encoder_path)
        else:
            self.use_cross_encoder = False

        if model is None:
            self.model = load()
        else:
            self.model = model
        self.model.summary()

    def __call__(self, claim_object, retrieved_abstracts):
        t1 = time.time()
        result = {}

        claim_id = claim_object["id"]
        claim_embedding = tf.constant(self.id_to_claim_embedding_map[claim_id])

        if self.number_of_abstracts_in_corpus == len(retrieved_abstracts):
            sentence_embeddings = self.sentence_embeddings
            rationale_id_to_abstract_and_sentence_id_pair = self.rationale_id_to_abstract_and_sentence_id_pair
        else:
            sentence_embeddings, rationale_id_to_abstract_and_sentence_id_pair = self.get_sentence_embeddings_for_retreived_abstracts(retrieved_abstracts)
        
        stacked_claim = np.ones((sentence_embeddings.shape[0], 1)) * claim_embedding
        datapoints = tf.concat([stacked_claim, sentence_embeddings], 1)
        predicted = tf.reshape(self.model(datapoints), (-1))

        results_above_threshold_mask = tf.squeeze(tf.math.greater(predicted, tf.constant(self.sbert_threshold)))
        indices_for_above_threshold = tf.where(results_above_threshold_mask)
        if indices_for_above_threshold.shape[0] == 0:
            return {}

        if self.use_cross_encoder:
            rationale_index_sorted_by_score = self.rerank_with_cross_encoder(
                claim_object,
                indices_for_above_threshold,
                rationale_id_to_abstract_and_sentence_id_pair
            )
        else:    
            rationale_index_sorted_by_score = self.sort_based_on_bi_encoder(
                indices_for_above_threshold,
                predicted
            )

        if self.k is not None:
            rationale_index_sorted_by_score = rationale_index_sorted_by_score[:self.k]

        for rationale_idx in rationale_index_sorted_by_score:
            abstract_id, sentence_id = rationale_id_to_abstract_and_sentence_id_pair[rationale_idx]
            abstract_rationales = result.setdefault(abstract_id, [])
            if len(abstract_rationales) < 3:
                abstract_rationales.append(sentence_id)
                result[abstract_id] = abstract_rationales
        t2 = time.time()
        print()
        print("TOTAL SENTENCE TIME:", t2-t1)
        return result

    def rerank_with_cross_encoder(self, claim_obj, indices_for_above_threshold, rationale_id_to_abstract_and_sentence_id_pair):
        cross_encoder_input = []
        for rationale_idx in indices_for_above_threshold:
            abstract_id, sentence_id = rationale_id_to_abstract_and_sentence_id_pair[rationale_idx[0]]
            sentence = self.id_to_abstract_map[abstract_id][sentence_id]
            cross_encoder_input.append((claim_obj["claim"], sentence))
        
        cross_scores = self.cross_encoder.predict(cross_encoder_input)

        rationale_index_and_score_pairs = [
            (rationale_idx[0], score) for rationale_idx, score in zip(indices_for_above_threshold, cross_scores)
        ]
        rationale_index_and_score_pairs_sorted_by_score = sorted(
            rationale_index_and_score_pairs, key=lambda tup: tup[1], reverse=True
        )
        return [idx for idx, score in rationale_index_and_score_pairs_sorted_by_score if score > self.cross_encoder_threshold]

    def sort_based_on_bi_encoder(self, indices_for_above_threshold, predicted):
        rationale_index_and_score_pairs = [
            (idx[0], predicted[idx[0]]) for idx in indices_for_above_threshold
        ]
        rationale_index_and_score_pairs_sorted_by_score = sorted(
            rationale_index_and_score_pairs, key=lambda tup: tup[1], reverse=True
        )
        return [idx for idx, _ in rationale_index_and_score_pairs_sorted_by_score]

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

    def get_sentence_embeddings_for_retreived_abstracts(self, retrieved_abstracts):
        rationale_id_to_abstract_and_sentence_id_pair = []
        sentence_embeddings = []

        for abstract_id in retrieved_abstracts.keys():
            abstract_embedding = self.id_to_abstract_embedding_map[abstract_id]
            for i in range(len(abstract_embedding)):
                rationale_id_to_abstract_and_sentence_id_pair.append((abstract_id, i))
            sentence_embeddings.append(np.array(abstract_embedding))

        sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
        return sentence_embeddings, rationale_id_to_abstract_and_sentence_id_pair

    def get_sentence_embeddings_for_all_abstracts(self, corpus_path):
        with jsonlines.open(corpus_path) as corpus_reader:
            corpus = np.array(list(corpus_reader.iter()))
        rationale_id_to_abstract_and_sentence_id_pair = []
        sentence_embeddings = []

        for line in corpus:
            for i in range(len(line["abstract"])):
                rationale_id_to_abstract_and_sentence_id_pair.append(
                    (line["doc_id"], i)
                )
            sentence_embeddings.append(np.array(line["abstract"]))

        sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
        return sentence_embeddings, rationale_id_to_abstract_and_sentence_id_pair

    def get_number_of_abstracts_in_corpus(self, corpus_path):
        with jsonlines.open(corpus_path) as corpus_reader:
            corpus = list(corpus_reader.iter())
            return len(corpus)


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
