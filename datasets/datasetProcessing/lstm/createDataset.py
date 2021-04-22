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


class DatasetType(enum.Enum):
    train = "train"
    validation = "validation"


class ScifactLSTMDataset:
    _set_type_to_directory = {
        DatasetType.train: "LstmTrainingDataset",
        DatasetType.validation: "LstmValidationDataset",
    }

    def __init__(self, set_type, claim_path="", corpus_path="", max_length=128):
        self.sequence_lenght = max_length

        directory = self._set_type_to_directory[set_type]
        path = (
            Path(os.path.realpath(__file__)).resolve().parents[1]
            / "tfrecords"
            / directory
        )
        if not os.path.exists(path):
            os.makedirs(path)

        self.dest = path / "scifact.tfrecord"
        self.corpus_path = corpus_path
        self.claim_path = claim_path

    def __call__(self):
        if not self._approve_overwriting():
            print("aborting dataset creation")
            return

        writer = tf.io.TFRecordWriter(str(self.dest))

        print("initializing tokenizer")
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
            add_special_tokens=True,
        )
        print("initializing id to abstact map")
        id_to_abstract_map = self._create_id_to_abstract_map()
        print("creating dataset and writeing to {}".format(self.dest))

        for claim_row in tqdm(jsonlines.open(self.claim_path)):
            claim, claim_mask = self._tokenize(claim_row["claim"])

            fallback = {id:{} for id in claim_row['cited_doc_ids']}
            evidence_set = claim_row["evidence"] if claim_row["evidence"] else fallback

            for abstract_id, rationales in evidence_set.items():
                abstract_obj = id_to_abstract_map[str(abstract_id)]
                abstract = abstract_obj["abstract"]
                title = abstract_obj["title"]

                rat_indices = [
                    sent_index
                    for rationale in rationales
                    for sent_index in rationale.get("sentences",[])
                ]  # a list of all the indices of the rationales

                for index in rat_indices:
                    sequence, sequence_mask = self._create_sequence(
                        claim, claim_mask, index, abstract, title
                    )
                    self._write(writer, sequence, sequence_mask, 1)

                negative_indice = [ind for ind in range(len(abstract)) if ind not in rat_indices]
                chosen_negative_indices = random.sample(
                    negative_indice, min(len(rat_indices) + 1, len(negative_indice)),
                )  # a list of indices of negative samples

                for index in chosen_negative_indices:
                    sequence, sequence_mask = self._create_sequence(
                        claim, claim_mask, index, abstract, title
                    )
                    self._write(writer, sequence, sequence_mask, 0)

        writer.flush()
        writer.close()

    def load(self):
        dataset = tf.data.TFRecordDataset(str(self.dest))
        return dataset.map(self._deserialize_example)

    """------------------------------- helper functions -------------------------------"""

    def _write(self, writer, sequence, sequence_mask, label):
        example = self._serialize_example(sequence, sequence_mask, label)
        writer.write(example)

    def _create_sequence(self, claim, claim_mask, sent_index, abstract, title):
        context, context_mask = (
            self._tokenize(title)
            if sent_index == 0
            else self._tokenize(abstract[sent_index - 1])
        )
        rationale, rationale_mask = self._tokenize(abstract[sent_index])
        sequence = tf.concat([claim, context, rationale], 0)
        sequence_mask = tf.concat([claim_mask, context_mask, rationale_mask], 0)
        return sequence, sequence_mask

    def _serialize_example(self, sequence, sequence_mask, label):
        features = {
            "claim": self._int64_feature(
                sequence[0]
            ),
            "context": self._int64_feature(
                sequence[1]
            ),
            "sentence": self._int64_feature(
                sequence[2]
            ),
            "claim_mask": self._int64_feature(
                sequence_mask[0]
            ),
            "context_mask": self._int64_feature(
                sequence_mask[1]
            ),
            "sentence_mask": self._int64_feature(
                sequence_mask[2]
            ),
            "label": self._int64_feature(
                [label]
            ),  # sequence of 0 and 1 denoting rationale sentences
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    def _deserialize_example(self, serialized_example):
        features = {
            "claim": tf.io.FixedLenFeature([self.sequence_lenght], tf.int64),
            "context": tf.io.FixedLenFeature([self.sequence_lenght], tf.int64),
            "sentence": tf.io.FixedLenFeature([self.sequence_lenght], tf.int64),
            "claim_mask": tf.io.FixedLenFeature([self.sequence_lenght], tf.int64),
            "context_mask": tf.io.FixedLenFeature([self.sequence_lenght], tf.int64),
            "sentence_mask": tf.io.FixedLenFeature([self.sequence_lenght], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        }

        example = tf.io.parse_single_example(serialized_example, features)
        claim = example["claim"]
        context = example["context"]
        sentence = example["sentence"]
        claim_mask = example["claim_mask"]
        context_mask = example["context_mask"]
        sentence_mask = example["sentence_mask"]
        return claim, context, sentence, claim_mask, context_mask, sentence_mask

    def _approve_overwriting(self):
        if os.path.exists(self.dest):
            choice = input(
                "you are about to overwrite one or more files? are you sure? [yes/no]\n"
            )
            if choice.lower().strip() in ["yes", "y", "ye"]:
                return True
            else:
                return False
        return True

    def _create_id_to_abstract_map(self):
        abstract_id_to_abstract = dict()
        corpus = jsonlines.open(self.corpus_path)
        for data in corpus:
            abstract_id_to_abstract[str(data["doc_id"])] = data

        return abstract_id_to_abstract

    def _tokenize(self, sentence):
        tokenization = list(self.tokenizer(
            sentence,
            return_attention_mask=True,
            return_tensors="tf",
            padding="max_length",
            max_length=self.sequence_lenght,
            truncation=True,
        ).values())
        return tokenization[0], tokenization[2]

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "claim_path",
        metavar="claim_path",
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
        "set_type", metavar="set_type", type=DatasetType, help="validation or train"
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
    t = ScifactLSTMDataset(
        args.set_type, args.claim_path, args.corpus_path, args.max_length
    )
    t()
    for claim, context, sentence, claim_m, context_m, sentence_m in t.load().take(1):
        print(claim, context, sentence)
        print("################")
        print(claim_m, context_m, sentence_m)

