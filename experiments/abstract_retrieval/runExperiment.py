""" Contains functionality to perform abstract retrieval """

import argparse
import jsonlines
import enum
from tqdm import tqdm
from utils import compute_f1, compute_precision, compute_recall
from dummyAbstractRetrieval import DummyAbstractRetrieval
from tfIdfAbstractRetrieval import TFIDFAbstractRetrieval
from bioSentVecAbstractRetrieval import BioSentVecAbstractRetrieval


def eval_abstract_retrieval(retriever, dataset_path):
    dataset = jsonlines.open(dataset_path)

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    for data in tqdm(dataset):
        claim = data["claim"]
        claim_id = data["id"]
        retrieved_abstracts = retriever.retrieve(claim_id, claim)
        gold_docs = data["evidence"].keys()

        true_positives = len(set(retrieved_abstracts).intersection(set(gold_docs)))
        false_positives = len(retrieved_abstracts) - true_positives
        false_negatives = len(gold_docs) - true_positives

        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

    precision = compute_precision(total_true_positives, total_false_positives)
    recall = compute_recall(total_true_positives, total_false_negatives)
    f1 = compute_f1(precision, recall)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)


class Retriever(enum.Enum):
    Dummy = "dummy"
    TFIDF = "tf-idf"
    BioSentVec = "BioSentVec"


if __name__ == "__main__":
    # Just for testing purpose
    parser = argparse.ArgumentParser(description="Create bar plots of evidence count")
    parser.add_argument(
        "data_set_path", metavar="path", type=str, help="the path to the dataset"
    )
    parser.add_argument("retriever", type=Retriever, metavar="retriever_func")
    parser.add_argument("-k", default=3, type=int)
    parser.add_argument(
        "-cl", "--claims_embedding", help="path to pickle file with claim embeddings"
    )
    parser.add_argument(
        "-co", "--corpus_embedding", metavar="path to pickle file with claim embeddings"
    )

    args = parser.parse_args()
    retriver = None
    if args.retriever == Retriever.Dummy:
        retriver = DummyAbstractRetrieval("{}/corpus.jsonl".format(args.data_set_path))
    elif args.retriever == Retriever.TFIDF:
        retriver = TFIDFAbstractRetrieval(
            args.k, "{}/corpus.jsonl".format(args.data_set_path), 1, 2
        )
    elif args.retriever == Retriever.BioSentVec:
        retriver = BioSentVecAbstractRetrieval(
            args.k, args.claims_embedding, args.corpus_embedding
        )

    eval_abstract_retrieval(retriver, "{}/claims_dev.jsonl".format(args.data_set_path))
