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
MODEL = None


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


def write_to_tf_record(writer, claim_embedding, relevance, label, *sentences):
    #assert len(claim_embedding) == 768
    for sentence in sentences:
        sentence_embedding = MODEL.encode(sentence).tolist()
        #assert len(sentence_embedding) == 768
        #concat = 
        #assert len(concat) == 1536
        writer.write(
            serialize_example(
                claim_embedding + sentence_embedding, relevance, label
            )
        )


def create_relevant(claim_path, corpus_path, set_type):
    id_to_abstact_map = create_id_to_abstract_map(corpus_path)
    directory = trainingset_path if set_type == DatasetType.train else validation_path
    writer = tf.io.TFRecordWriter(str(directory / "scifact_relevant.tfrecord"))
    for claim in tqdm(jsonlines.open(claim_path)):
        if not claim["evidence"]:
            continue

        claim_embedding = MODEL.encode(claim["claim"]).tolist()
        sentences = []
        # paper says that all the abstracts chosen agrees on the label, so we just use the label of the first evidence
        label = 1 if list(claim["evidence"].values())[0][0]["label"] == "SUPPORT" else 0
        for doc_id, evidence_sets in claim["evidence"].items():
            encoded_abstract = id_to_abstact_map[doc_id]
            usefull = []
            for indexes in [evidence["sentences"] for evidence in evidence_sets]:
                usefull.extend(indexes)
            usefill_sentences = [encoded_abstract[index] for index in usefull]
            sentences.extend(usefill_sentences)

        write_to_tf_record(writer, claim_embedding, 1, label, *sentences)


def create_not_relevant(claim_path, corpus_path, k, set_type):
    id_to_abstact_map = create_id_to_abstract_map(corpus_path)
    directory = trainingset_path if set_type == DatasetType.train else validation_path
    writer = tf.io.TFRecordWriter(str(directory / "scifact_not_relevant.tfrecord"))
    for claim in tqdm(jsonlines.open(claim_path)):
        claim_embedding = MODEL.encode(claim["claim"]).tolist()
        negative_abstracts = [
            abstract
            for doc_id, abstract in id_to_abstact_map.items()
            if doc_id not in claim["evidence"]
        ]

        # make not_relecant datapoint from random abstract not used for evidence
        negative_sentences = []
        for abstract in sample(negative_abstracts, k):
            chosen_sentences = sample(abstract, 1)
            negative_sentences.extend(chosen_sentences)
        write_to_tf_record(writer, claim_embedding, 0, 0, *negative_sentences)

        evidence_obj = claim["evidence"]
        if not evidence_obj:
            continue
        # make not_relevant for the abstrac with gold rationales
        for doc_id, evidence in evidence_obj.items():
            abstract = id_to_abstact_map[doc_id]
            not_allowed = []
            for obj in evidence:
                not_allowed.extend(obj["sentences"])
            not_allowed = set(not_allowed)
            allowed = [
                abstract[index] for index in set(range(0, len(abstract))) - not_allowed
            ]
            chosen = sample(allowed, min(2, len(allowed)))
            write_to_tf_record(writer, claim_embedding, 0, 0, *chosen)

    # flush and close()
    writer.flush()
    writer.close()


def create_fever_relevant(claim_path, set_type):
    directory = trainingset_path if set_type == DatasetType.train else validation_path
    relevant_writer = tf.io.TFRecordWriter(str(directory / "fever_relevant.tfrecord"))
    not_relevant_writer = tf.io.TFRecordWriter(
        str(directory / "fever_not_relevant.tfrecord")
    )

    for claim in tqdm(jsonlines.open(claim_path)):
        claim_embedding = MODEL.encode(claim["claim"]).tolist()
        # create a not relevant datapoint if not enough info
        if claim["label"] == "NOT ENOUGH INFO":
            allowed = [x for x in claim["sentences"] if x != ""]
            negative_evidence = sample(allowed, min(len(allowed), 2))
            write_to_tf_record(
                not_relevant_writer, claim_embedding, 0, 0, *negative_evidence
            )
            continue

        # clean sentences by removing the part with the references.
        cleaned_sentences = list(
            map(lambda a: a.split(" . ")[0] + ".", claim["sentences"])
        )
        # set label to 1 if the claim is supported by abstract, else 0
        label = 1 if claim["label"] == "SUPPORTS" else 0
        # else create a relevant and not relevant datapoints from the sentence used to label
        for evidence in claim["evidence_sets"]:
            all_evidence = set()
            for index in evidence:
                all_evidence = all_evidence.union(set(evidence))

            # write to the relevant dataset
            usefull = [cleaned_sentences[index] for index in evidence]
            write_to_tf_record(relevant_writer, claim_embedding, 1, label, *usefull)

            # write to the not_relevant dataset
            not_relevant_sents = set(range(len(cleaned_sentences))) - set(all_evidence)
            allowed = [
                cleaned_sentences[index]
                for index in not_relevant_sents
                if cleaned_sentences[index] != ""
            ]
            chosen = sample(allowed, min(2, len(allowed)))
            write_to_tf_record(not_relevant_writer, claim_embedding, 0, 0, *chosen)

    # flush and close files
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
        "set_type", metavar="type", type=DatasetType, help="validation or train"
    )
    parser.add_argument(
        "-f", "--fever", action="store_true", help="is it a fever dataset or not"
    )
    parser.add_argument(
        "-c",
        "--corpus_path",
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
    parser.add_argument(
        "-r",
        "--relevance",
        metavar="relevance",
        type=Relevancy,
        default=Relevancy.relevant,
        help="choose between relevant and not relevant",
    )

    args = parser.parse_args()
    MODEL = SentenceTransformer("stsb-distilbert-base")

    if args.fever:
        create_fever_relevant(args.claim_path, args.set_type)
    elif args.relevance == Relevancy.relevant:
        create_relevant(args.claim_path, args.corpus_path, args.set_type)
    elif args.relevance == Relevancy.not_relevant:
        create_not_relevant(args.claim_path, args.corpus_path, args.k, args.set_type)
