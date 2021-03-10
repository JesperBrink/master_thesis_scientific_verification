""" Helper script for analysis of using BioSentVec for abstract retrieval """

import jsonlines
import argparse
from experiments.abstract_retrieval.bioSentVecAbstractRetrieval import (
    BioSentVecAbstractRetrieval,
)
from tqdm import tqdm


def load_abstract_dict(corpus_path):
    abstract_id_to_abstract = dict()
    corpus = jsonlines.open(corpus_path)
    for data in corpus:
        abstract_id_to_abstract[str(data["doc_id"])] = data["abstract"]

    return abstract_id_to_abstract


def main(dev_path, corpus_path, claims_embedding, corpus_embedding, interactive):
    abstract_id_to_abstract = load_abstract_dict(corpus_path)

    dev_claims_dataset = jsonlines.open(dev_path)
    retriever = BioSentVecAbstractRetrieval(
        3,
        claims_embedding,
        corpus_embedding,
    )

    total_length = 0
    not_enough_info_counter = 0

    for data in tqdm(dev_claims_dataset):
        total_length += 1
        claim_id = data["id"]
        retrieved_abstracts = retriever.retrieve(claim_id, None)
        gold_docs = data["evidence"].keys()

        true_positives = len(set(retrieved_abstracts).intersection(set(gold_docs)))
        false_negatives = len(gold_docs) - true_positives

        if len(gold_docs) == 0:
            not_enough_info_counter += 1

        if interactive and false_negatives > 0:
            print("\n#" * 40)
            print("Num of false negatives:", false_negatives)
            print("Claim Id:", claim_id)
            print("\t" + data["claim"])

            print("Gold docs:")
            for gold_abstract in gold_docs:
                print("\t" + gold_abstract + ":")
                for i, sentence in enumerate(abstract_id_to_abstract[gold_abstract]):
                    print("\t\t" + str(i) + ": " + str(sentence))
                print("\tSentence overview:", data["evidence"][gold_abstract])
            print()

            print("Retrieved abstracts:")
            for abstract in retrieved_abstracts:
                print("\t" + abstract + ": " + str(abstract_id_to_abstract[abstract]))

            input("Pres enter to continue")

    print(
        "Number of abstracts with 0 evidence: "
        + str(not_enough_info_counter)
        + "/"
        + str(total_length)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyse the performance of BioSentVec on abstract retrieval"
    )
    parser.add_argument("claim_path", help="the path to the jsonl file with claims")
    parser.add_argument("corpus_path", help="path to the corpus file")
    parser.add_argument(
        "claims_embedding", help="path to pickle file with claim embeddings"
    )
    parser.add_argument(
        "corpus_embedding", help="path to pickle file with claim embeddings"
    )
    parser.add_argument(
        "-i",
        "--interactive",
        help="Set this flag if you want in interactive session",
        action="store_true",
    )

    args = parser.parse_args()
    main(
        args.claim_path,
        args.corpus_path,
        args.claims_embedding,
        args.corpus_embedding,
        args.interactive,
    )
