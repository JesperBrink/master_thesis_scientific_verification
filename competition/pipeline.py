import jsonlines
import numpy as np
from tqdm import tqdm
import models.sentence_selection.model as sentence_selection_module
from models.sentence_selection.cosine_similarity_model import CosineSimilaritySentenceSelector
from models.filter_corpus.cosine_similarity import CosineSimilarityFilterModel
from models.filter_corpus.bm25 import BM25FilterModel
import models.stance_prediction.model as stance_prediction_module
import time
import tensorflow as tf
import enum
import argparse


def sentence_selection(claim, model, sentence_embeddings, corp_id, threshold):
    """Returns a dict that maps abstract ids to relevant sentences ids in that abstract
    i.e. {abstract_42: [{id: sent_3, embedding: <embedding>}, {id: sent_7, embedding: <embedding>}], abstract_127: [...]}
    We have at most 9 sentences per abstract
    """
    claim = tf.ones((sentence_embeddings.shape[0], 1)) * claim["claim"]
    claim_sent_embedding = tf.concat([claim, tf.cast(sentence_embeddings, tf.float32)], 1)

    predicted = model(claim_sent_embedding)
    res_mask = tf.squeeze(tf.math.greater(predicted, tf.constant(threshold)))
    res = tf.where(res_mask)    
    
    relevant_sentences_dict = dict()
    for pred_id in res:
        pred_id_val = pred_id[0]
        abstract_id, sentence_id = corp_id[pred_id_val]
        sentence_list = relevant_sentences_dict.get(abstract_id, [])
        score = predicted[pred_id_val][0]
        sentence_list.append((score, sentence_id, pred_id_val))
        relevant_sentences_dict[abstract_id] = sentence_list

    # Only use best 3 sentences
    for abstract_id, sentence_list in relevant_sentences_dict.items():
        sorted_by_score = sorted(sentence_list, key=lambda tup: tup[0], reverse=True)
        top_3_sentences_by_score = sorted_by_score[:3]
        correct_format = [{"id": sentence_id, "embedding": claim_sent_embedding[pred_id_val]} for _, sentence_id, pred_id_val in top_3_sentences_by_score]
        relevant_sentences_dict[abstract_id] = correct_format
    
    return relevant_sentences_dict


def same_prediction_as_avg(avg, pred, threshold):
    if avg < threshold and pred < threshold:
        return True
    elif avg >= threshold and pred >= threshold:
        return True
    return False


def stance_prediction(claim, evidence, model, output_writer=None):
    """
    input: Claims + Rationales (from sentence selection)
    output: Whether abstracts/sentences support or refute claims
    """
    claim_id = claim["id"]

    if not evidence:
        return {"id": claim_id, "evidence": {}}

    resulting_evidence_dict = dict()
    for abstract in evidence.keys():
        stance_predictions = []
        pred_sum = 0

        for sentence_dict in evidence[abstract]:
            embedding = tf.expand_dims(sentence_dict["embedding"], 0)
            pred = model(embedding)
            stance_predictions.append((sentence_dict["id"], pred))
            pred_sum += pred

        avg = pred_sum / len(stance_predictions)

        if output_writer is not None:
            output_writer.write("AVG predicted label value: {}".format(avg))

        threshold = tf.constant(0.5)
        rationale_sentences = [
            sent_id
            for sent_id, pred in stance_predictions
            if same_prediction_as_avg(avg, pred, threshold)
        ]
        label = "SUPPORT" if avg >= threshold else "CONTRADICT"
        resulting_evidence_dict[str(abstract)] = {
            "sentences": rationale_sentences,
            "label": label,
        }

    return {"id": claim_id, "evidence": resulting_evidence_dict}


def setup_sentence_embeddings(corpus_path):
    with jsonlines.open(corpus_path) as corpus_reader:
        corpus = np.array(list(corpus_reader.iter()))
    corp_id = []
    sentence_embeddings = []
    
    for line in corpus:
        for i in range(len(line['abstract'])):
            corp_id.append((line['doc_id'], i))
        sentence_embeddings.append(np.array(line['abstract']))

    sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
    return sentence_embeddings, corp_id


def run_pipeline(corpus_path, claims_path, sentence_selection_model, stance_prediction_model, filter_model):
    threshold = 0.5
    
    sentence_embeddings, corp_id = setup_sentence_embeddings(corpus_path)
    with jsonlines.open("predictions.jsonl", "w") as output:
        with jsonlines.open(claims_path) as claims:
            for claim in tqdm(claims):
                if filter_model is not None:
                    sentence_embeddings = filter_model.get_top_k_by_similarity(claim, sentence_embeddings, 50)

                relevant_sentences_dict = sentence_selection(claim, sentence_selection_model, sentence_embeddings, corp_id, threshold)
                prediction = stance_prediction(claim, relevant_sentences_dict, stance_prediction_model) 
                output.write(prediction)


class FilterModel(enum.Enum):
    NONE = "none"
    SBERT_COSINE_SIMILARITY = "cosine"
    BM25 = "bm25" # separate bm25, i.e. bm25-sent and bm25-abstract?

class SentenceSelctionModel(enum.Enum):
    TWO_LAYER_DENSE = "twolayer"
    SBERT_COSINE_SIMILARITY = "cosine"

class StancePredictionModel(enum.Enum):
    TWO_LAYER_DENSE = "twolayer"


if __name__ == "__main__":
    #corpus_path = "sbert-embedded-corpus.jsonl"
    #claims_path = "sbert-embedded-dev-claims.jsonl"
    corpus_path = "../stsb-distilbert-base-finetuned-scifact100-embedded-corpus.jsonl"
    claims_path = "../stsb-distilbert-base-finetuned-scifact100-embedded-dev-claims.jsonl"

    parser = argparse.ArgumentParser(
        description="Script to run evaluation pipeline"
    )
    parser.add_argument(
        "filter_model",
        metavar="filter",
        type=FilterModel,
        help="Which pruning model to use. none = No pruning, cosine = SBERT + cosine similarity, bm25 = BM25",
    )
    parser.add_argument(
        "sentence_selection_model",
        metavar="sentence_selection_model",
        type=SentenceSelctionModel,
        help="Which sentence selection model to use. twolayer = Two layer dense, cosine = SBERT cosine similarity",
    )
    parser.add_argument(
        "stance_prediction_model",
        metavar="stance_prediction_model",
        type=StancePredictionModel,
        help="Which stance prediction model to use. twolayer = Two layer dense",
    )

    args = parser.parse_args()

    if args.filter_model == FilterModel.NONE:
        filter_model = None
    elif args.filter_model == FilterModel.SBERT_COSINE_SIMILARITY:
        filter_model = CosineSimilarityFilterModel()
    elif args.filter_model == FilterModel.BM25:
        filter_model = BM25FilterModel(corpus_path)
    else:
        raise NotImplementedError()

    if args.sentence_selection_model == SentenceSelctionModel.TWO_LAYER_DENSE:
        sentence_selection_model = sentence_selection_module.load()
    elif args.sentence_selection_model == SentenceSelctionModel.SBERT_COSINE_SIMILARITY:
        sentence_selection_model = CosineSimilaritySentenceSelector()
    else:
        raise NotImplementedError()

    if args.stance_prediction_model == StancePredictionModel.TWO_LAYER_DENSE:
        stance_prediction_model = stance_prediction_module.load()
    else:
        raise NotImplementedError()

    run_pipeline(corpus_path, claims_path, sentence_selection_model, stance_prediction_model, filter_model)
