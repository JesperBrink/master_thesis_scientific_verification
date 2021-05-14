import argparse
from datetime import datetime
import jsonlines
import os
from pathlib import Path

import tensorflow as tf
from transformers import BertConfig, TFBertModel, BertTokenizer

from datasets.datasetProcessing.base.baseModelContextDataset import load
from models.utils import get_highest_count

_model_dir = (
    Path(os.path.realpath(__file__)).resolve().parents[1]
    / "trained_models/sentence_selection/base"
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

MAX_LENGTH = 196

class BaseModelWithContextSelector:
    def __init__(self, corpus_path, threshold=0.5):
        self.threshold = threshold
        self.model = load_model()
        self.model.summary()
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
            add_special_tokens=True,
        )
        self.id_to_title = self._create_id_to_title_map(corpus_path)

    def __call__(self, claim_object, abstracts):
        claim = claim_object["claim"]

        result = {}
        for doc_id, sents in abstracts.items():
            title = self.id_to_title[doc_id]
            sents_with_title = [title] + sents
            temp = zip(sents_with_title, sents)
            tokenized_input = []
            tokenized_mask = []
            for context, sent in temp:
                res = self.tokenizer(
                    claim,
                    context + " " + sent,
                    return_attention_mask=True,
                    return_tensors="tf",
                    padding="max_length",
                    max_length=MAX_LENGTH,
                    truncation=True,
                )

                tokenized_input.append(res.input_ids[0])
                tokenized_mask.append(res.attention_mask[0])

            model_result = tf.reshape(
                self.model([tf.stack(tokenized_input), tf.stack(tokenized_mask)]), (-1)
            )

            top_k, indices = tf.math.top_k(model_result, k=3)
            res = tf.reshape(
                tf.gather(indices, tf.where(top_k > self.threshold), axis=0), (-1)
            )
            rationales = res.numpy().tolist()
            if len(rationales) < 1:
                continue
            result[doc_id] = rationales

        return result

    def _create_id_to_title_map(self, path):
        abstract_id_to_title = dict()
        corpus = jsonlines.open(path)
        for data in corpus:
            abstract_id_to_title[data["doc_id"]] = data["title"]
        return abstract_id_to_title


def check_for_folder():
    if not _model_dir.exists():
        print("Creating save folder")
        os.makedirs(_model_dir)


def base_model_with_context_selector(
    bert_dropout, attention_dropout, classification_dropout
):
    check_for_folder()

    model_name = "bert-base-uncased"
    config = BertConfig.from_pretrained(model_name)
    config.dropout = bert_dropout
    config.attention_dropout = attention_dropout
    config.output_hidden_states = False
    bert_embedding = TFBertModel.from_pretrained(
        model_name, config=config, name="bert"
    ).bert
    bert_embedding.trainable = False

    def make_trainable():
        bert_embedding.trainable = True

    sequence_input = tf.keras.Input(shape=(MAX_LENGTH,), dtype="int32", name="sequence")
    sequence_mask = tf.keras.Input(shape=(MAX_LENGTH,), dtype="int32", name="claim_mask")
    # print(sequence_input.shape)
    bert_encoding = bert_embedding(
        input_ids=sequence_input, attention_mask=sequence_mask
    )[1]
    # print(bert_encoding.shape)
    dropout = tf.keras.layers.Dropout(
        classification_dropout, name="classification_dropout"
    )(bert_encoding)
    # print(dropout.shape)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout)
    # print(output.shape)

    model = tf.keras.Model(
        inputs=[sequence_input, sequence_mask],
        outputs=output,
        name="bert-lstm-sentence-selection",
    )

    model.summary()

    return model, make_trainable


def train(model, make_trainable, loss, batch_size, frozen_epochs, epochs, bert_lr):
    print("Start training")
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    train, val = load()

    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(logs, update_freq=1)
    model.fit(
        train.batch(batch_size),
        epochs=frozen_epochs,
        shuffle=True,
        validation_data=val.batch(batch_size),
        callbacks=[tb_callback],
    )
    save(model, "frozen_")
    print(
        "initial training of classification is done\nNow finetuning bert model to task"
    )

    make_trainable()

    opt = tf.keras.optimizers.Adam(learning_rate=bert_lr)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    model.fit(
        train.batch(batch_size),
        epochs=epochs,
        shuffle=True,
        validation_data=val.batch(batch_size),
        callbacks=[tb_callback],
    )


def load_model(prefix=""):
    count = get_highest_count(_model_dir)
    path = str(
        _model_dir / "{}base_with_context_abstract_retriver_{}".format(prefix, count)
    )
    print("Loading model from: {}".format(str(path)))
    model = tf.keras.models.load_model(path)
    return model


def save(model, prefix=""):
    count = get_highest_count(_model_dir) + 1
    path = str(
        _model_dir / "{}base_with_context_abstract_retriver_{}".format(prefix, count)
    )
    model.save(path)
    print("model saved to {}".format(path))


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t", "--train", action="store_true", help="will train the model if set"
    )

    parser.add_argument(
        "-w",
        "--work",
        action="store_true",
        help="will run a small test of the evaluator. Can be used to test load and senetence selection",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=10, help="the number of epochs"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, help="the batch_size", default=16
    )
    parser.add_argument("-fe", "--frozen_epochs", type=int, default=10)
    parser.add_argument("-co", "--corpus_embedding", type=str)
    parser.add_argument("-ad", "--attention_dropout", type=float, default=0.1)
    parser.add_argument("-bd", "--bert_dropout", type=float, default=0.1)
    parser.add_argument("-cd", "--classification_dropout", type=float, default=0.1)
    parser.add_argument("-bl", "--bert_learningrate", type=float, default=0.00001)

    args = parser.parse_args()
    if args.train:
        model, make_trainable = base_model_with_context_selector(
            args.bert_dropout, args.attention_dropout, args.classification_dropout
        )
        loss = tf.keras.losses.BinaryCrossentropy()
        train(
            model,
            make_trainable,
            loss,
            args.batch_size,
            args.frozen_epochs,
            args.epochs,
            args.bert_learningrate,
        )
        save(model)
    if args.work:
        # model = load_model()
        # model.summary()
        selector = BaseModelWithContextSelector(args.corpus_embedding, 0.00)
        abstracts = {
            4983: [
                "ID elements are short interspersed elements (SINEs) found in high copy number in many rodent genomes.",
                "BC1 RNA, an ID-related transcript, is derived from the single copy BC1 RNA gene.",
                "The BC1 RNA gene has been shown to be a master gene for ID element amplification in rodent genomes.",
                "ID elements are dispersed through a process termed retroposition.",
                "The retroposition process involves a number of potential regulatory steps.",
                "These regulatory steps may include transcription in the appropriate tissue, transcript stability, priming of the RNA transcript for reverse transcription and integration.",
                "This study focuses on priming of the RNA transcript for reverse transcription.",
                "BC1 RNA gene transcripts are shown to be able to prime their own reverse transcription in an efficient intramolecular and site-specific fashion.",
                "This self-priming ability is a consequence of the secondary structure of the 3'-unique region.",
                "The observation that a gene actively amplified throughout rodent evolution makes a RNA capable of efficient self-primed reverse transcription strongly suggests that self-priming is at least one feature establishing the BC1 RNA gene as a master gene for amplification of ID elements.",
            ]
        }
        print(selector({"id": 14, "claim": "gd is not"}, abstracts))


if __name__ == "__main__":
    main()
