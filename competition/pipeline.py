import jsonlines
import numpy as np
from tqdm import tqdm
from models.sentence_selection.baseModelWithContext import BaseModelWithContextSelector
from models.sentence_selection.lstmModel import BertLSTMSentenceSelector
from models.sentence_selection.denseModel import TwoLayerDenseSentenceSelector
from models.sentence_selection.cosineSimilarityModel import (
    CosineSimilaritySentenceSelector,
)
from models.stance_prediction.denseModel import TwoLayerDenseStancePredictor
from models.abstract_retriever.tf_idf import TFIDFAbstractRetrieval
from models.abstract_retriever.bm25 import BM25AbstractRetrieval
import time
import tensorflow as tf
import enum
import argparse
import torch
import time

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

class NoopAbstractRetriever:
    def __call__(self, claim, abstracts):
        return abstracts


class DevAbstractRetriever:
    def __call__(self, claim, abstracts):
        return {x: y for x, y in list(abstracts.items())[:10]}


class DevSentenceSelector:
    def __call__(self, claim, abstracts):
        for doc_id, _ in abstracts.items():
            return {doc_id: [0, 1]}


def create_id_to_abstract_map(corpus_path):
    abstract_id_to_abstract = dict()
    corpus = jsonlines.open(corpus_path)
    for data in corpus:
        abstract_id_to_abstract[data["doc_id"]] = data["abstract"]

    return abstract_id_to_abstract


def pipeline(
    claims_path, corpus_path, abstract_retriever, sentence_selector, stance_predictor
):
    abstracts = create_id_to_abstract_map(corpus_path)
    abstract_retrieval_times = []
    rationale_selection_times = []
    stance_prediction_times = []
    claim_ids = []
    with jsonlines.open("predictions.jsonl", "w") as output_writer:
        with jsonlines.open(claims_path) as claims:
            for claim_object in tqdm(claims):
                t1 = time.time()
                retrieved_abstracts = abstract_retriever(claim_object, abstracts)
                t2 = time.time()
                selected_sentences = sentence_selector(
                    claim_object, retrieved_abstracts
                )
                t3 = time.time()
                prediction = stance_predictor(
                    claim_object, selected_sentences, retrieved_abstracts
                )
                t4 = time.time()
                abstract_retrieval_times.append(t2-t1)
                rationale_selection_times.append(t3-t2)
                stance_prediction_times.append(t4-t3)
                claim_ids.append(claim_object["id"])
                output_writer.write(prediction)
   
    with open("abstract_retrieval_times", "w") as abstract_retrieval_writer:
        for claim_id, t in zip(claim_ids, abstract_retrieval_times):
            abstract_retrieval_writer.write("claim id: {}, time: {}\n".format(claim_id, t))
        abstract_retrieval_writer.write("\n")
        abstract_retrieval_writer.write("AVG: {}\n".format(str(np.mean(abstract_retrieval_times))))
        abstract_retrieval_writer.write("VAR: {}\n".format(str(np.var(abstract_retrieval_times))))
        abstract_retrieval_writer.write("MAX: {}\n".format(str(np.max(abstract_retrieval_times))))
        abstract_retrieval_writer.write("MIN: {}\n".format(str(np.min(abstract_retrieval_times))))

    with open("rationale_selection_times", "w") as rationale_selection_writer:
        for claim_id, t in zip(claim_ids, rationale_selection_times):
            rationale_selection_writer.write("claim id: {}, time: {}\n".format(claim_id, t))
        rationale_selection_writer.write("\n")
        rationale_selection_writer.write("AVG: {}\n".format(str(np.mean(rationale_selection_times))))
        rationale_selection_writer.write("VAR: {}\n".format(str(np.var(rationale_selection_times))))
        rationale_selection_writer.write("MAX: {}\n".format(str(np.max(rationale_selection_times))))
        rationale_selection_writer.write("MIN: {}\n".format(str(np.min(rationale_selection_times))))

    with open("stance_prediction_times", "w") as stance_prediction_writer:
        for claim_id, t in zip(claim_ids, stance_prediction_times):
            stance_prediction_writer.write("claim id: {}, time: {}\n".format(claim_id, t))
        stance_prediction_writer.write("\n")
        stance_prediction_writer.write("AVG: {}\n".format(str(np.mean(stance_prediction_times))))
        stance_prediction_writer.write("VAR: {}\n".format(str(np.var(stance_prediction_times))))
        stance_prediction_writer.write("MAX: {}\n".format(str(np.max(stance_prediction_times))))
        stance_prediction_writer.write("MIN: {}\n".format(str(np.min(stance_prediction_times))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run evaluation pipeline")
    parser.add_argument(
        "abstract_retriever",
        metavar="abstract_retriever",
        type=str,
        choices=["dev", "noop", "tfidf", "bm25"],
        help="Which model to use for abstract retrieval. dev = quick testing, noop = No pruning, tfid = TF-IDF, bm25 = Pyserini BM25",
    )
    parser.add_argument(
        "sentence_selector",
        metavar="sentence_selector",
        type=str,
        choices=["dev", "lstm", "dense", "dense_rerank", "cosine", "cosine_rerank", "base"],
        help="Which sentence selection model to use. dev = for quick testing, lstm = bert-lstm model, dense = Two layer dense, cosine = SBERT cosine similarity",
    )
    parser.add_argument(
        "stance_predictor",
        metavar="stance_predictor",
        type=str,
        choices=["dense"],
        help="Which stance prediction model to use. dense = Two layer dense",
    )
    parser.add_argument(
        "claim_path", metavar="path", type=str, help="the path to the sentence claims"
    )
    parser.add_argument(
        "corpus_path",
        metavar="path",
        type=str,
        help="the path to the sentence corpus",
    )
    parser.add_argument(
        "-st",
        "--sentence_threshold",
        type=float,
        default=0.5,
        help="the threshold for sentence selection",
    )
    parser.add_argument(
        "-ct",
        "--cross_encoder_threshold",
        type=float,
        default=0.5,
        help="the threshold for the cross encoder used for sentence selection",
    )
    parser.add_argument(
        "-cp",
        "--cross_encoder_path",
        type=str,
        help="the path to or name of the cross encoder used for sentence selection",
    )
    parser.add_argument(
        "-k",
        "--overall_top_k",
        type=int,
        help="how many rationales the rationale selection should return at most",
    )
    parser.add_argument("-cl", "--claim_embedding", type=str)
    parser.add_argument("-co", "--corpus_embedding", type=str)

    args = parser.parse_args()

    if args.abstract_retriever == "noop":
        abstract_retriever = NoopAbstractRetriever()
    elif args.abstract_retriever == "dev":
        abstract_retriever = DevAbstractRetriever()
    elif args.abstract_retriever == "tfidf":
        abstract_retriever = TFIDFAbstractRetrieval(args.corpus_path)
    elif args.abstract_retriever == "bm25":
        abstract_retriever = BM25AbstractRetrieval(args.corpus_path, k=3)

    if args.sentence_selector == "lstm":
        sentence_selector = BertLSTMSentenceSelector(
            args.corpus_path, args.sentence_threshold
        )
    elif args.sentence_selector == "dev":
        sentence_selector = DevSentenceSelector()
    elif args.sentence_selector == "dense":
        sentence_selector = TwoLayerDenseSentenceSelector(
            args.corpus_embedding,
            args.claim_embedding,
            sbert_threshold=args.sentence_threshold,
            k=args.overall_top_k,
        )
    elif args.sentence_selector == "dense_rerank":
        sentence_selector = TwoLayerDenseSentenceSelector(
            args.corpus_embedding,
            args.claim_embedding,
            sbert_threshold=args.sentence_threshold,
            k=args.overall_top_k,
            cross_encoder_threshold=args.cross_encoder_threshold,
            cross_encoder_path=args.cross_encoder_path,
            corpus_path=args.corpus_path
        )
    elif args.sentence_selector == "cosine":
        sentence_selector = CosineSimilaritySentenceSelector(
            args.corpus_embedding,
            args.claim_embedding,
            sbert_threshold=args.sentence_threshold,
            k=args.overall_top_k,
        )
    elif args.sentence_selector == "cosine_rerank":
        sentence_selector = CosineSimilaritySentenceSelector(
            args.corpus_embedding,
            args.claim_embedding,
            sbert_threshold=args.sentence_threshold,
            k=args.overall_top_k,
            cross_encoder_threshold=args.cross_encoder_threshold,
            cross_encoder_path=args.cross_encoder_path,
            corpus_path=args.corpus_path
        )
    elif args.sentence_selector == "base":
        sentence_selector = BaseModelWithContextSelector(
            args.corpus_path, args.sentence_threshold
        )

    if args.stance_predictor == "dense":
        stance_predictor = TwoLayerDenseStancePredictor(
            args.corpus_embedding, args.claim_embedding
        )

    pipeline(
        args.claim_path,
        args.corpus_path,
        abstract_retriever,
        sentence_selector,
        stance_predictor,
    )
