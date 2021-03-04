import argparse
import jsonlines
import enum
import os
import tensorflow as tf
from random import sample
import csv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np

# paths to the folder where the
trainingset_path = (
    Path(os.path.realpath(__file__)).resolve().parents[1] / "trainingDataset"
)
validation_path = (
    Path(os.path.realpath(__file__)).resolve().parents[1] / "validationDataset"
)

# s-bert model name
model_name = "stsb-distilbert-base"


def create_id_to_abstract_map(corpus_path):
    abstract_id_to_abstract = dict()
    corpus = jsonlines.open(corpus_path)
    for data in corpus:
        abstract_id_to_abstract[str(data["doc_id"])] = data["abstract"]

    return abstract_id_to_abstract


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(inp, relevance, label):
    feature = {
        "X": _float_feature(
            inp
        ),  # the embeded claim concatinated witht the embedded sentence
        "relevance": _int64_feature(
            relevance
        ),  # 0 if not relevant (Not enough info) 1 of refute or support
        "label": _int64_feature(label),  # 0 if Refuting 1 if supporting
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_relevant(claim_path, corpus_path, set_type):
    model = SentenceTransformer(model_name)
    id_to_abstact_map = create_id_to_abstract_map(corpus_path)
    directory = trainingset_path if set_type == DatasetType.train else validation_path
    with tf.io.TFRecordWriter(str(directory / "scifact_relevant.tfrecord")) as writer:
        for claim in tqdm(jsonlines.open(claim_path)):
            if claim["evidence"]:
                claim_encoding = model.encode(claim["claim"])
                for doc_id, evidence_sets in claim["evidence"].items():
                    encoded_abstract = id_to_abstact_map[doc_id]
                    # set label to 1 if the claim is supported by abstract, else 0
                    label = 1 if evidence_sets["label"] == "SUPPORT" else 0
                    for evidence in model(evidence_sets).tolist():
                        [
                            writer.write(
                                serialize_example(
                                    claim_encoding.tolist() + encoded_abstract[index],
                                    1,
                                    label,
                                )
                            )
                            for index in evidence["sentences"]
                        ]


def create_not_relevant(claim_path, corpus_path, k, set_type):
    model = SentenceTransformer(model_name)
    id_to_abstact_map = create_id_to_abstract_map(corpus_path)
    directory = trainingset_path if set_type == DatasetType.train else validation_path
    with tf.io.TFRecordWriter(
        str(directory / "scifact_not_relevant.tfrecord")
    ) as writer:
        for claim in tqdm(jsonlines.open(claim_path)):
            claim_encoding = model.encode(claim["claim"])
            temp_list = [
                value
                for key, value in id_to_abstact_map.items()
                if key not in claim["evidence"]
            ]
            for abstract in sample(temp_list, k):
                sentence = sample(abstract, 1)
                for sent in sentence:
                    writer.write(
                        serialize_example(
                            claim_encoding.tolist() + model(sent).tolist(), 0, 0
                        )
                    )
            evidence_obj = claim["evidence"]
            if not evidence_obj:
                continue
            for doc_id, evidence in evidence_obj.items():
                abstract = id_to_abstact_map[doc_id]
                not_allowed = []
                for obj in evidence:
                    not_allowed.extend(obj["sentences"])
                not_allowed = set(not_allowed)
                allowed = set(range(0, len(abstract))) - not_allowed
                chosen = sample(allowed, 1)[0]
                writer.write(
                    serialize_example(
                        claim_encoding.tolist() + model(abstract[chosen]).tolist(), 0, 0
                    )
                )


def create_fever_relevant(claim_path, set_type):
    model = SentenceTransformer(model_name)
    directory = trainingset_path if set_type == DatasetType.train else validation_path
    relevant_writer = tf.io.TFRecordWriter(
        str(directory / "fever_relevant.tfrecord")
    )
    not_relevant_writer = tf.io.TFRecordWriter(
        str(directory / "fever_not_relevant.tfrecord")
    )

    for claim in tqdm(jsonlines.open(claim_path)):
        claim_encoding = model.encode(claim["claim"])
        # create a not relevant datapoint if not enough info
        if claim["label"] == "NOT ENOUGH INFO":
            negative_evidence = sample(claim["sentences"], 1)[0]
            not_relevant_writer.write(
                serialize_example(
                    claim_encoding.tolist()
                    + model.encode(negative_evidence).tolist(),
                    0,
                    0,
                )
            )
            continue
        # else create a relevant datapoint
        cleaned_sentences = list(
            map(lambda a: a.split(" . ")[0] + ".", claim["sentences"])
        )
        # set label to 1 if the claim is supported by abstract, else 0
        label = 1 if claim["label"] == "SUPPORTS" else 0
        for evidence in claim["evidence_sets"]:
            for index in evidence:
                relevant_writer.write(
                    serialize_example(
                        claim_encoding.tolist()
                        + model.encode(cleaned_sentences[index]).tolist(),
                        1,
                        label,
                    )
                )

    relevant_writer.flush()
    relevant_writer.close()
    not_relevant_writer.flush()
    not_relevant_writer.close()


class Relevancy(enum.Enum):
    relevant = "relevant"
    not_relevant = "notrelevant"


class DatasetType(enum.Enum):
    train = "train"
    validation = "validation"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "claim_path", metavar="path", type=str, help="the path to the sentence claims"
    )
    parser.add_argument(
        "relevance",
        metavar="relevance",
        type=Relevancy,
        help="choose between relevant and not relevant",
    )
    parser.add_argument(
        "set_type", metavar="type", type=DatasetType, help="validation or train"
    )
    parser.add_argument(
        "-f", "--fever", action="store_true", help="is it a fever dataset or not"
    )
    parser.add_argument(
        "-c",
        "-corpus_path",
        metavar="path",
        type=str,
        help="the path to the sentence corpus",
    )
    parser.add_argument(
        "-k",
        metavar="k",
        type=int,
        help="the number of not relevant sentence pr claim",
        default=5,
    )

    args = parser.parse_args()
    if args.relevance == Relevancy.relevant and args.fever:
        create_fever_relevant(args.claim_path, args.set_type)
    elif args.relevance == Relevancy.relevant:
        create_relevant(args.claim_path, args.corpus_path, args.type)
    elif args.relevance == Relevancy.not_relevant:
        create_not_relevant(args.claim_path, args.corpus_path, args.k, args.type)
