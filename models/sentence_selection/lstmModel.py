import os
from pathlib import Path
import argparse
from datetime import datetime
import jsonlines
import tensorflow as tf
from transformers import TFBertModel, BertConfig, BertTokenizer
from tqdm import tqdm
from models.utils import get_highest_count, setup_tensorboard
from datasets.datasetProcessing.lstm.createDataset import (
    ScifactLSTMDataset,
    DatasetType,
)

_model_dir = (
    Path(os.path.realpath(__file__)).resolve().parents[1]
    / "trained_models/sentence_selection/lstm"
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class BertLSTMSentenceSelector:
    def __init__(self, corpus_path, threshold=0.5):
        self.threshold = threshold
        self.model = load()
        self.model.summary()
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
            add_special_tokens=True,
        )
        self.id_to_title = self._create_id_to_title_map(corpus_path)

    def __call__(self, claim_object, abstracts):
        claim = claim_object["claim"]
        claim_token, claim_attention_mask = self._tokenize(claim)

        result = {}
        for doc_id, sents in abstracts.items():
            title_encoding, title_mask = self._tokenize(self.id_to_title[doc_id])
            abstract_encoding, abstract_mask = self._tokenize(sents)
            abstract_text = tf.concat([title_encoding, abstract_encoding], 0)
            abstract_text_mask = tf.concat([title_mask, abstract_mask], 0)

            # Create input for model
            claim_input = tf.tile(claim_token, [len(abstract_text) - 1, 1])
            context_input = abstract_text[:-1]
            sent_input = abstract_text[1:]

            # create attention masks
            claim_mask = tf.tile(claim_attention_mask, [len(abstract_text_mask) - 1, 1])
            context_mask = abstract_text_mask[:-1]
            sent_mask = abstract_text_mask[1:]

            model_result = tf.reshape(
                self.model(
                    [
                        claim_input,
                        context_input,
                        sent_input,
                        claim_mask,
                        context_mask,
                        sent_mask,
                    ],
                    
                ),
                (-1),
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

    def _tokenize(self, sentence):
        tokenization = list(
            self.tokenizer(
                sentence,
                return_attention_mask=True,
                return_tensors="tf",
                padding="max_length",
                max_length=128,
                truncation=True,
            ).values()
        )
        return tokenization[0], tokenization[2]

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


def lstm_abstract_retriever(
    units,
    attention_dropout,
    bert_dropout,
    lstm_dropout,
    classification_dropout,
    bert_trainable=False,
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

    # INPUT
    claim = tf.keras.Input(shape=(128,), dtype="int32", name="claim")
    context = tf.keras.Input(shape=(128,), dtype="int32", name="context")
    sentence = tf.keras.Input(shape=(128,), dtype="int32", name="sentence")
    claim_mask = tf.keras.Input(shape=(128,), dtype="int32", name="claim_mask")
    context_mask = tf.keras.Input(shape=(128,), dtype="int32", name="context_mask")
    sentence_mask = tf.keras.Input(shape=(128,), dtype="int32", name="sentence_mask")

    # ENCODING
    claim_embedding = bert_embedding(input_ids=claim, attention_mask=claim_mask)[0][
        :, 0, :
    ]
    context_embedding = bert_embedding(input_ids=context, attention_mask=context_mask)[
        0
    ][:, 0, :]
    sent_embedding = bert_embedding(input_ids=sentence, attention_mask=sentence_mask)[
        0
    ][:, 0, :]

    # SEQUALISE
    concat = tf.keras.layers.Concatenate(axis=1)(
        [claim_embedding, context_embedding, sent_embedding]
    )
    reshape = tf.keras.layers.Reshape((3, 768))(concat)

    # LSTM
    lstm_dropout_layer = tf.keras.layers.Dropout(lstm_dropout, name="lstm_dropout")(
        reshape
    )
    lstm = tf.keras.layers.LSTM(
        units, return_sequences=False, recurrent_initializer="glorot_uniform"
    )(lstm_dropout_layer)

    # CLASSIFICATION
    classification_dropout_layer = tf.keras.layers.Dropout(
        classification_dropout, name="classification_dropout"
    )(lstm)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(
        classification_dropout_layer
    )

    model = tf.keras.Model(
        inputs=[claim, context, sentence, claim_mask, context_mask, sentence_mask],
        outputs=outputs,
        name="bert-lstm-sentence-selection",
    )

    model.summary()

    return model, make_trainable


def train(
    model,
    loss,
    flip_function,
    frozen_epochs=10,
    epochs=10,
    batch_size=16,
    shuffle=True,
    bert_lr=0.00001,
):
    print("Start training")
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    train = ScifactLSTMDataset(DatasetType.train).load().batch(batch_size)
    val = ScifactLSTMDataset(DatasetType.validation).load().batch(batch_size)

    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(logs, update_freq=1)
    model.fit(
        train,
        epochs=frozen_epochs,
        shuffle=shuffle,
        validation_data=val,
        callbacks=[tb_callback],  # ,
    )
    save(model, "frozen_")
    print("initial training of lstm is done\nNow finetuning bert model to task")

    flip_function()

    opt = tf.keras.optimizers.Adam(learning_rate=bert_lr)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    model.fit(
        train,
        epochs=epochs,
        shuffle=shuffle,
        validation_data=val,
        callbacks=[tb_callback],  # ,
    )


def load():
    count = get_highest_count(_model_dir)
    path = str(_model_dir / "bert_lstm_abstract_retriver_{}".format(count))
    print("Loading model from: {}".format(str(path)))
    model = tf.keras.models.load_model(path)
    return model


def save(model, prefix=""):
    count = get_highest_count(_model_dir) + 1
    path = str(_model_dir / "{}bert_lstm_abstract_retriver_{}".format(prefix, count))
    model.save(path)
    print("model saved to {}".format(path))


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t", "--train", action="store_true", help="will train the model if set"
    )
    parser.add_argument(
        "-u",
        "--lstm_units",
        type=int,
        help="The number of units in the lstm layer",
        default=512,
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, help="the batch_size", default=16
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
    parser.add_argument("-fe", "--frozen_epochs", type=int, default=10)
    parser.add_argument("-co", "--corpus_embedding", type=str)
    parser.add_argument("-ad", "--attention_dropout", type=float, default=0.1)
    parser.add_argument("-bd", "--bert_dropout", type=float, default=0.1)
    parser.add_argument("-ld", "--lstm_dropout", type=float, default=0.1)
    parser.add_argument("-cd", "--classification_dropout", type=float, default=0.1)
    parser.add_argument("-bl", "--bert_learningrate", type=float, default=0.00001)


    args = parser.parse_args()

    if args.train:
        m, flip_function = lstm_abstract_retriever(
            args.lstm_units,
            args.attention_dropout,
            args.bert_dropout,
            args.lstm_dropout,
            args.classification_dropout,
        )
        loss = tf.keras.losses.BinaryCrossentropy()
        train(
            m,
            loss,
            flip_function,
            batch_size=args.batch_size,
            epochs=args.epochs,
            frozen_epochs=args.frozen_epochs,
            bert_lr=args.bert_learningrate,
        )
        save(m)
    if args.work:
        selector = BertLSTMSentenceSelector(args.corpus_embedding, 0.00)
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
