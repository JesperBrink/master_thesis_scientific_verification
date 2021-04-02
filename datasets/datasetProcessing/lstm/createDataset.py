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
from transformers import DistilBertTokenizer


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
        self.dest = (
            Path(os.path.realpath(__file__)).resolve().parents[1]
            / directory
            / "scifact.tfrecord"
        )
       
        self.corpus_path = corpus_path
        self.claim_path = claim_path

    def __call__(self):
        writer = tf.io.TFRecordWriter(str(self.dest))
        if not self._approve_overwriting():
            print("aborting dataset creation")
            return

        print("initializing tokenizer")
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased",
            do_lower_case=True,
            add_special_tokens=True,
        )
        print("initializing id to abstact map")
        id_to_abstract_map = self._create_id_to_abstract_map()
        print("creating dataset and writeing to {}".format(self.dest))
        zeroes = tf.zeros((1,self.sequence_lenght), dtype=tf.dtypes.int32)

        for claim in tqdm(jsonlines.open(self.claim_path)):
            tokenized_claim, claim_mask = self._tokenize(claim['claim'])
            evidence_set = claim.get("evidence", {})
            for abstract_id, rationales in evidence_set.items():
                abstract = id_to_abstract_map[str(abstract_id)]

                rat_indices = [
                    sent_index
                    for rationale in rationales
                    for sent_index in rationale["sentences"]
                ]  # a list of all the indices of the rationales

                for index in rat_indices:
                    context, context_mask = (zeroes, zeroes) if index == 0 else self._tokenize(abstract[index-1])
                    rationale, rationale_mask = self._tokenize(abstract[index])
                    sequence = tf.concat([tokenized_claim, context, rationale], 0)
                    sequence_mask = tf.concat([claim_mask, context_mask, rationale_mask], 0)
                    self._write(writer, sequence, sequence_mask, 1)
            
            # TODO: Create negative examples
        
        writer.flush()
        writer.close()
    
    def load(self):
        dataset = tf.data.TFRecordDataset(str(self.dest))
        return dataset.map(self._deserialize_example)

    """------------------------------- helper functions -------------------------------"""

    def _write(self, writer, sequence, sequence_mask, label):
        example = self._serialize_example(sequence, sequence_mask, label)
        writer.write(example)


    def _serialize_example(self, sequence, sequence_mask, label):
        features = {
            "sequence": self._int64_feature(
                tf.reshape(sequence, [-1])
            ),  # flattened tensor for wordpieces
            "sequence_mask": self._int64_feature(tf.reshape(sequence_mask, [-1])), # flattened tensor of mask matching sequence 
            "label": self._int64_feature(
                [label]
            ),  # sequence of 0 and 1 denoting rationale sentences
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    def _deserialize_example(self, serialized_example):
        features = {
            "sequence": tf.io.FixedLenFeature([3, self.sequence_lenght], tf.int64),
            "sequence_mask": tf.io.FixedLenFeature([3, self.sequence_lenght], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.int64)
        }

        example = tf.io.parse_single_example(serialized_example, features)
        X = (example["sequence"], example["sequence_mask"])
        Y = example["label"]
        return X, Y


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
            abstract_id_to_abstract[str(data["doc_id"])] = data["abstract"]

        return abstract_id_to_abstract

    def _tokenize(self, sentence):
        return self.tokenizer(
            sentence,
            return_attention_mask=True,
            return_tensors="tf",
            padding="max_length",
            max_length=self.sequence_lenght,
            truncation=True,
        ).values()

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
    )()