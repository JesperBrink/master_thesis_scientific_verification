import pickle
import os

from sentence_transformers import evaluation


def load_training_data(data_folder_path):
    scifact_training_data = pickle.load(open(os.path.join(data_folder_path, "scifact_training_data.p"), "rb"))
    fever_training_data = pickle.load(open(os.path.join(data_folder_path, "fever_training_data.p"), "rb"))

    return scifact_training_data, fever_training_data
    

def load_evaluator(data_folder_path):
    claims = pickle.load(open(os.path.join(data_folder_path, "claims.p"), "rb"))
    sentences = pickle.load(open(os.path.join(data_folder_path, "sentences.p"), "rb"))
    labels = pickle.load(open(os.path.join(data_folder_path, "labels.p"), "rb"))

    return evaluation.EmbeddingSimilarityEvaluator(claims, sentences, labels)
