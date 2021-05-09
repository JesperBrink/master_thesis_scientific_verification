import argparse
import enum
import jsonlines
from pathlib import Path
import os
from datetime import datetime as time
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer

train_path = (
    Path(os.path.realpath(__file__)).resolve().parents[1]
    / "tfrecords"
    / "BaseModelContextTrainSet"
)
val_path = (
    Path(os.path.realpath(__file__)).resolve().parents[1]
    / "tfrecords"
    / "BaseModelContextValidationSet"
)


class BaseModelContextDataset:
    def __init__(self, corpus_path):
        self.validationDatasetPath = val_path
        self.trainDatasetPath = train_path

        if not os.path.exists(self.validationDatasetPath):
            os.makedirs(self.validationDatasetPath)
        if not os.path.exists(self.trainDatasetPath):
            os.makedirs(self.trainDatasetPath)

        print("initializing tokenizer")
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
            add_special_tokens=True,
        )
        print("initializing id to abstact map")
        self.id_to_abstract_map = self._create_id_to_abstract_map(corpus_path)

    def __call__(self, validation_claim, train_claim):
        validation_dest = self.validationDatasetPath / "scifact.tfrecord"
        train_dest = self.trainDatasetPath / "scifact.tfrecord"

        if not self._approve_overwriting(validation_dest, train_dest):
            print("aborting dataset creation")
            return

        print("Processing validation")
        validation_writer = tf.io.TFRecordWriter(str(validation_dest))
        self._process_claims(validation_writer, validation_claim)

        print("Processing train")
        train_writer = tf.io.TFRecordWriter(str(train_dest))
        self._process_claims(train_writer, train_claim)

    def _process_claims(self, writer, claim_path):
        for claim_row in tqdm(jsonlines.open(claim_path)):
            claim = claim_row["claim"]

            fallback = {id: {} for id in claim_row["cited_doc_ids"]}
            evidence_set = claim_row["evidence"] if claim_row["evidence"] else fallback

            for abstract_id, rationales in evidence_set.items():
                abstract_obj = self.id_to_abstract_map[str(abstract_id)]
                abstract = abstract_obj["abstract"]
                title = abstract_obj["title"]

                rat_indices = [
                    sent_index
                    for rationale in rationales
                    for sent_index in rationale.get("sentences", [])
                ]  # a list of all the indices of the rationales

                for index in rat_indices:
                    context = title if index == 0 else abstract[index - 1]
                    sent = abstract[index]
                    sequence, sequence_mask = self._create_sequence(
                        claim, context, sent
                    )
                    self._write(writer, sequence, sequence_mask, 1)

                negative_indice = [
                    ind for ind in range(len(abstract)) if ind not in rat_indices
                ]
                chosen_negative_indices = random.sample(
                    negative_indice,
                    min(len(rat_indices) + 1, len(negative_indice)),
                )  # a list of indices of negative samples

                for index in chosen_negative_indices:
                    context = title if index == 0 else abstract[index - 1]
                    sent = abstract[index]
                    sequence, sequence_mask = self._create_sequence(
                        claim, context, sent
                    )
                    self._write(writer, sequence, sequence_mask, 0)

        writer.flush()
        writer.close()

    def _create_sequence(self, claim, context, sent):
        combined = context + " " + sent
        res = self.tokenizer(
            claim,
            combined,
            return_attention_mask=True,
            return_tensors="tf",
            padding="max_length",
            max_length=256,
            truncation=True,
        )
        return res.input_ids[0], res.attention_mask[0]

    def _create_id_to_abstract_map(self, corpus_path):
        abstract_id_to_abstract = dict()
        corpus = jsonlines.open(corpus_path)
        for data in corpus:
            abstract_id_to_abstract[str(data["doc_id"])] = data

        return abstract_id_to_abstract

    def _approve_overwriting(self, *dest):
        if any([os.path.exists(str(d)) for d in dest]):
            choice = input(
                "you are about to overwrite one or more files? are you sure? [yes/no]\n"
            )
            if choice.lower().strip() in ["yes", "y", "ye"]:
                return True
            else:
                return False
        return True

    def _write(self, writer, sequence, sequence_mask, label):
        example = self._serialize_example(sequence, sequence_mask, label)
        writer.write(example)

    def _serialize_example(self, sequence, sequence_mask, label):
        features = {
            "sequence": self._int64_feature(sequence),
            "sequence_mask": self._int64_feature(sequence_mask),
            "label": self._int64_feature(
                [label]
            ),  # sequence of 0 and 1 denoting rationale sentences
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _deserialize_example(serialized_example):
    features = {
        "sequence": tf.io.FixedLenFeature([256], tf.int64),
        "sequence_mask": tf.io.FixedLenFeature([256], tf.int64),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }

    example = tf.io.parse_single_example(serialized_example, features)
    sequence = example["sequence"]
    sequence_mask = example["sequence_mask"]
    label = example["label"]
    return (sequence, sequence_mask), label


def _load_dataset(source):
    dataset = tf.data.TFRecordDataset(str(source))
    return dataset.map(_deserialize_example)


def load():
    train = _load_dataset(train_path / "scifact.tfrecord")
    val = _load_dataset(val_path / "scifact.tfrecord")
    return train, val


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "train_path",
        metavar="train_path",
        type=str,
        help="the path to the sentence claims",
    )
    parser.add_argument(
        "val_path",
        metavar="val_path",
        type=str,
        help="the path to the sentence claims",
    )
    parser.add_argument(
        "corpus_path",
        metavar="corpus_path",
        type=str,
        help="the path to the sentence corpus",
    )
    parser.add_argument(
        "-l",
        "--max_length",
        metavar="max_length",
        type=int,
        help="the number of not relevant sentence pr claim",
        default=128,
    )

    args = parser.parse_args()
    # BaseModelContextDataset(
    # args.corpus_path
    # )(args.val_path, args.train_path)

    train, val = load()
    print("here")
    for x, y in train.take(10):
        print(x)


if __name__ == "__main__":
    main()
