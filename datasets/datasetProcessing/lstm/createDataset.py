import argparse
import enum
import jsonlines
from pathlib import Path
import os
from datetime import datetime as time

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

trainingset_path = (
    Path(os.path.realpath(__file__)).resolve().parents[1] / "LstmTrainingDataset"
)
validation_path = (
    Path(os.path.realpath(__file__)).resolve().parents[1] / "LstmValidationDataset"
)

def approve_overwriting(path, *file_names):
    if any([os.path.exists(path / file_name) for file_name in file_names]):
        choice = input(
            "you are about to overwrite one or more files? are you sure? [yes/no]\n"
        )
        if choice.lower().strip() in ["yes", "y", "ye"]:
            return True
        else:
            return False
    return True

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
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(inp, label):
    feature = {
        "sequence": _float_feature(inp), # the flattende matrix of the abstract
        "label": _int64_feature(label),  # sequence of 0 and 1 denoting rationale sentences 
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_to_tf_record(writer, claim_embedding, label_sequence, abstract, model):
    claim_embedding_matrix = np.full((len(abstract),len(claim_embedding)), claim_embedding)
    abstract_embedding_matrix = model.encode(abstract)
    sequence = np.concatenate((claim_embedding_matrix, abstract_embedding_matrix), axis=1)
    flattened_sequence = sequence.flatten()
    writer.write(serialize_example(flattened_sequence, label_sequence))
   

def create_scifact_dataset(claim_path, corpus_path, model, dataset_type):
    id_to_abstact_map = create_id_to_abstract_map(corpus_path)
    directory = trainingset_path if dataset_type == DatasetType.train else validation_path
    file_name = "scifact.tfrecord"

    if not approve_overwriting(directory, file_name):
        print("aborting dataset creation")
        return

    writer = tf.io.TFRecordWriter(str(directory / file_name))
    for claim in tqdm(jsonlines.open(claim_path)):
        claim_embedding = model.encode(claim["claim"]).tolist()    
        evidence_set = claim["evidence"]
        if evidence_set:
            for abstract_id, rationales in evidence_set.items():
                rationale_indices = []
                for indices in [rationale['sentences'] for rationale in rationales]:
                    rationale_indices.extend(indices)
                abstract = id_to_abstact_map[str(abstract_id)]
                classification_sequence = [1 if x in rationale_indices else 0 for x in range(0, len(abstract))]
                write_to_tf_record(writer, claim_embedding, classification_sequence, abstract, model)
            continue

        # Has no evidence use cited_doc_ids
        for doc_id in claim["cited_doc_ids"]:
            abstract = id_to_abstact_map[str(doc_id)]
            zeroes = [0 for _ in range(len(abstract))]
            write_to_tf_record(writer, claim_embedding, zeroes, abstract, model)

    writer.flush()
    writer.close()


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

    args = parser.parse_args()
    print("initializing model")
    model= SentenceTransformer("stsb-distilbert-base")
    print("creating dataset")
    if args.fever:
        print("f is set")
    elif args.corpus_path:
        create_scifact_dataset(args.claim_path, args.corpus_path, model, args.set_type)
    else:
        print("Argument Error: use -f if its a feverdataset, else the -c argument is needed")