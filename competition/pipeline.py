import jsonlines
import numpy as np
from tqdm import tqdm
# import models.sentence_selection.model as sentence_selection_module
# from models.sentence_selection.cosine_similarity_model import CosineSimilaritySentenceSelector
# from models.filter_corpus.cosine_similarity import CosineSimilarityFilterModel
# from models.filter_corpus.bm25 import BM25FilterModel
from models.sentence_selection.lstmModel import BertLSTMSentenceSelector
from models.sentence_selection.denseModel import TwoLayerDenseSentenceSelector
from models.stance_prediction.model import TwoLayerDenseSrancePredictor
from models.abstract_retriever.tf_idf import TFIDFAbstractRetrieval
import models.stance_prediction.model as stance_prediction_module
import time
import tensorflow as tf
import enum
import argparse

class NoopAbstractRetriever():
    def __call__(self, claim, abstracts):
        return abstracts

class DevAbstractRetriever():
    def __call__(self, claim, abstracts):
        return {x:y for x, y in list(abstracts.items())[:10]}

class DevSentenceSelector():
    def __call__(self, claim, abstracts):
        for doc_id, _ in abstracts.items():
            return {doc_id: [0,1]}

def create_id_to_abstract_map(corpus_path):
    abstract_id_to_abstract = dict()
    corpus = jsonlines.open(corpus_path)
    for data in corpus:
        abstract_id_to_abstract[data["doc_id"]] = data["abstract"]

    return abstract_id_to_abstract

def pipeline(claims_path, corpus_path, abstract_retriever, sentence_selector, stance_predictor):
    abstracts = create_id_to_abstract_map(corpus_path)
    with jsonlines.open("predictions.jsonl", "w") as output_writer:
        with jsonlines.open(claims_path) as claims:
            for claim_object in tqdm(claims):
                retrieved_abstracts = abstract_retriever(claim_object, abstracts)
                selected_sentences = sentence_selector(claim_object, retrieved_abstracts)
                prediction = stance_predictor(claim_object, selected_sentences, retrieved_abstracts)
                output_writer.write(prediction) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run evaluation pipeline"
    )
    parser.add_argument(
        "abstract_retriever",
        metavar="abstract_retriever",
        type=str,
        choices=["dev", "noop", "tfidf", "bm25"],
        help="Which model to use for abstract retrieval. none = No pruning, cosine = SBERT + cosine similarity, bm25 = BM25",
    )
    parser.add_argument(
        "sentence_selector",
        metavar="sentence_selector",
        type=str,
        choices=["dev", "lstm", "dense"],
        help="Which sentence selection model to use. dense = Two layer dense, cosine = SBERT cosine similarity",
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
        "-st", "--sentence_threshold", type=float, default=0.5,
        help="the threshold for sentence selection"
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

    if args.abstract_retriever == "noop":
        abstract_retriever = NoopAbstractRetriever()
    elif args.abstract_retriever == "dev":
        abstract_retriever = DevAbstractRetriever()
    elif args.abstract_retriever == "tfidf":
        abstract_retriever = TFIDFAbstractRetrieval(args.corpus_path)

    if args.sentence_selector == "lstm":
        sentence_selector = BertLSTMSentenceSelector(args.sentence_threshold)
    elif args.sentence_selector == "dev":
        sentence_selector = DevSentenceSelector()
    elif args.sentence_selector == "dense":
        sentence_selector = TwoLayerDenseSentenceSelector(args.corpus_embedding, args.claim_embedding)

    if args.stance_predictor == "dense":
        stance_predictor = TwoLayerDenseSrancePredictor(args.corpus_embedding, args.claim_embedding)
    
    pipeline(args.claim_path, args.corpus_path, abstract_retriever, sentence_selector, stance_predictor)
