""" Contains functionality to perform abstract retrieval """

import argparse
import jsonlines
from utils import compute_f1, compute_precision, compute_recall
from dummyAbstractRetrieval import DummyAbstractRetrieval


def eval_abstract_retrieval(retriever, dataset_path):
    dataset = jsonlines.open(dataset_path)

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    for data in dataset:
        claim = data["claim"]
        retrieved_abstracts = retriever.retrieve(claim)
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


if __name__ == "__main__":
    # Just for testing purpose
    parser = argparse.ArgumentParser(description="Create bar plots of evidence count")
    parser.add_argument(
        "data_set_path", metavar="path", type=str, help="the path to the dataset"
    )

    args = parser.parse_args()
    dummy = DummyAbstractRetrieval("{}/corpus.jsonl".format(args.data_set_path))
    eval_abstract_retrieval(dummy, "{}/claims_dev.jsonl".format(args.data_set_path))
