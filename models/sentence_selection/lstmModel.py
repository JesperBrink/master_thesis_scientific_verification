import os
from pathlib import Path
import argparse
from datetime import datetime

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
    def __init__(self, corpus_paht, threshold=0.5):
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
            abstract_text = [title_encoding]
            abstract_text_masks = [title_mask]
            for sent in sents:
                text, mask = self._tokenize(sent)
                abstract_text.append(text)
                abstract_text_masks.append(mask)
            repeat = tf.repeat([abstract_text, abstract_text_masks], repeats=2, axis=1)
            sequence = tf.reshape(repeat[:, 1:-1],(2, -1, 2, 128))
            tiled_claim = tf.reshape(
                tf.tile([claim_token, claim_attention_mask], [1, len(sents), 1]),
                (2, -1, 1, 128),
            )
            inputs, inputs_mask = tf.concat([tiled_claim, sequence], 2)
            model_result = tf.reshape(self.model([inputs, inputs_mask]), (-1))
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
        corpus = jsonlines.open(paht)
        for data in corpus:
            abstract_id_to_title[str(data["doc_id"])] = data["title"]
        return abstract_id_to_title

def check_for_folder():
    if not _model_dir.exists():
        print("Creating save folder")
        os.makedirs(_model_dir)


def lstm_abstract_retriever(units):
    check_for_folder()
    
    config = BertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    bert_embedding = TFBertModel.from_pretrained(
        "bert-base-uncased", config=config, name="bert"
    ).bert
    bert_embedding.trainable = False
    
    inputs = tf.keras.Input(shape=(3, 128), dtype="int32", name="sequence")
    inputs_mask = tf.keras.Input(shape=(3, 128), dtype="int32", name="attention_masks")
    print(inputs_mask.shape)
    claim_embedding = bert_embedding(inputs[:,0], attention_mask=inputs_mask[:,0])[1]
    context_embedding = bert_embedding(inputs[:,1], attention_mask=inputs_mask[:,1])[1]
    sent_embedding = bert_embedding(inputs[:,2], attention_mask=inputs_mask[:,2])[1]
    print(sent_embedding.shape)
    concat = tf.keras.layers.Concatenate()([claim_embedding, context_embedding, sent_embedding])
    print(concat.shape)
    reshape = tf.keras.layers.Reshape((3, 768))(concat)
    print(reshape.shape)
    lstm = tf.keras.layers.LSTM(
        units, return_sequences=False, recurrent_initializer="glorot_uniform"
    )(reshape)
    print(lstm.shape)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(lstm)
    print(outputs.shape)

    model = tf.keras.Model(
        inputs=[inputs, inputs_mask],
        outputs=outputs,
        name="bert-lstm-sentence-selection",
    )

    model.summary()

    return model


def train(model, epochs=10, batch_size=16, shuffle=True):
    train = ScifactLSTMDataset(DatasetType.train).load().batch(batch_size)
    val = ScifactLSTMDataset(DatasetType.validation).load().batch(batch_size)

    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    model.fit(
        train,
        epochs=epochs,
        shuffle=shuffle,
        validation_data=val,
    )


def load():
    count = get_highest_count(_model_dir)
    path = str(_model_dir / "bert_lstm_abstract_retriver_{}".format(count))
    print("Loading model from: {}".format(str(path)))
    model = tf.keras.models.load_model(path)
    return model


def save(model):
    count = get_highest_count(_model_dir) + 1
    path = str(_model_dir / "bert_lstm_abstract_retriver_{}".format(count))
    model.save(path)
    print("model saved to {}".format(path))


if __name__ == "__main__":
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
        "-e", "--epochs", type=int, help="the number of epochs", default=10
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
    parser.add_argument("-co", "--corpus_embedding", type=str)
    args = parser.parse_args()

    if args.train:
        m = lstm_abstract_retriever(args.lstm_units)
        loss = tf.keras.losses.BinaryCrossentropy()
        m.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        train(m, batch_size=args.batch_size, epochs=args.epochs)
        save(m)
        m = load()
        # m.summary()
    if args.work:
        selector = BertLSTMSentenceSelector(args.corpus_embedding, 0.00)
        abstracts = {
            4983:[
                'ID elements are short interspersed elements (SINEs) found in high copy number in many rodent genomes.',
                'BC1 RNA, an ID-related transcript, is derived from the single copy BC1 RNA gene.', 
                'The BC1 RNA gene has been shown to be a master gene for ID element amplification in rodent genomes.', 
                'ID elements are dispersed through a process termed retroposition.', 
                'The retroposition process involves a number of potential regulatory steps.', 
                'These regulatory steps may include transcription in the appropriate tissue, transcript stability, priming of the RNA transcript for reverse transcription and integration.', 
                'This study focuses on priming of the RNA transcript for reverse transcription.', 
                'BC1 RNA gene transcripts are shown to be able to prime their own reverse transcription in an efficient intramolecular and site-specific fashion.', 
                "This self-priming ability is a consequence of the secondary structure of the 3'-unique region.", 
                'The observation that a gene actively amplified throughout rodent evolution makes a RNA capable of efficient self-primed reverse transcription strongly suggests that self-priming is at least one feature establishing the BC1 RNA gene as a master gene for amplification of ID elements.']
        }
        print(selector({"id":14,"claim":"gd is not"}, abstracts))
