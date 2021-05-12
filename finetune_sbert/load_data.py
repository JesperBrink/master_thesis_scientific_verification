import pickle
import os

from sentence_transformers import evaluation
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator,
)


def load_training_data(data_folder_path):
    scifact_training_data = pickle.load(
        open(os.path.join(data_folder_path, "scifact_training_data.p"), "rb")
    )
    fever_training_data = pickle.load(
        open(os.path.join(data_folder_path, "fever_training_data.p"), "rb")
    )

    return scifact_training_data, fever_training_data


def load_evaluator(data_folder_path):
    claims = pickle.load(open(os.path.join(data_folder_path, "claims.p"), "rb"))
    sentences = pickle.load(open(os.path.join(data_folder_path, "sentences.p"), "rb"))
    labels = pickle.load(open(os.path.join(data_folder_path, "labels.p"), "rb"))

    return evaluation.EmbeddingSimilarityEvaluator(claims, sentences, labels)


def load_cross_encoder_evaluator(data_folder_path):
    claims = pickle.load(open(os.path.join(data_folder_path, "claims.p"), "rb"))
    sentences = pickle.load(open(os.path.join(data_folder_path, "sentences.p"), "rb"))
    labels = pickle.load(open(os.path.join(data_folder_path, "labels.p"), "rb"))

    claim_sentence_pairs = [
        [claim, sentence] for claim, sentence in zip(claims, sentences)
    ]

    return CEBinaryClassificationEvaluator(claim_sentence_pairs, labels)
