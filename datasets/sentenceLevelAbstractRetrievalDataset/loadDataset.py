import argparse
import jsonlines
import tensorflow as tf
import csv
from sentence_transformers import SentenceTransformer


def create_id_to_abstract_map(corpus_path):
    abstract_id_to_abstract = dict()
    corpus = jsonlines.open(corpus_path)
    for data in corpus:
        abstract_id_to_abstract[str(data["doc_id"])] = data["abstract"]

    return abstract_id_to_abstract


def create_relevant(claim_path, corpus_path, output_path):
    model = SentenceTransformer("stsb-distilbert-base")
    id_to_abstact_map = create_id_to_abstract_map(corpus_path)
    dest = open(output_path, "w")
    csv_writer = csv.writer(dest)

    for claim in jsonlines.open(claim_path):
        if claim["evidence"]:
            claim_encoding = model.encode(claim["claim"])
            for doc_id, evidence_sets in claim["evidence"].items():
                encoded_abstract = id_to_abstact_map[doc_id]
                for evidence in evidence_sets:
                    [
                        csv_writer.writerow(
                            [claim_encoding.tolist() + encoded_abstract[index], 1]
                        )
                        for index in evidence["sentences"]
                    ]

    dest.close()


def create_not_relevant(claim_path, corpus_path, output_path):
    pass


def main(claim_path, corpus_path, output_path):
    create_relevant(claim_path, corpus_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "claim_path", metavar="path", type=str, help="the path to the claims"
    )
    parser.add_argument(
        "corpus_path", metavar="path", type=str, help="the path to the corpus"
    )
    parser.add_argument(
        "output_path", metavar="path", type=str, help="the path to the output file"
    )

    args = parser.parse_args()
    main(args.claim_path, args.corpus_path, args.output_path)
