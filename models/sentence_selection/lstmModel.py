import os
from pathlib import Path
import argparse
from datetime import datetime

import tensorflow as tf
from transformers import TFBertModel, BertConfig

from models.utils import get_highest_count, setup_tensorboard
from datasets.datasetProcessing.lstm.createDataset import (
    ScifactLSTMDataset,
    DatasetType,
)

_model_dir = (
    Path(os.path.realpath(__file__)).resolve().parents[1]
    / "trained_models/sentence_selection"
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def lstm_abstract_retriever(untis):
    config = BertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    bert_embedding = TFBertModel.from_pretrained(
        "bert-base-uncased", config=config, name="bert"
    ).bert

    inputs = tf.keras.Input(shape=(3, 128), dtype="int32", name="sequence")
    # print(inputs.shape)
    inputs_mask = tf.keras.Input(shape=(3, 128), dtype="int32", name="attention_masks")
    # print(inputs_mask.shape)
    reshaped_inputs = tf.reshape(inputs, [-1, 128], name="seq_reshape")
    # print(reshaped_inputs.shape)
    reshaped_mask = tf.reshape(inputs_mask, [-1, 128], name="seq_mask_reshape")
    # print(reshaped_mask.shape)
    pooling_embedding = bert_embedding(reshaped_inputs, attention_mask=reshaped_mask)[1]
    # print(pooling_embedding.shape)
    reshaped_cls = tf.reshape(pooling_embedding, [-1, 3, 768], name="cls_reshape")
    # print(reshaped_cls.shape)
    lstm = tf.keras.layers.LSTM(
        untis, return_sequences=False, recurrent_initializer="glorot_uniform"
    )(reshaped_cls)
    # print(lstm.shape)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(lstm)
    # print(outputs.shape)

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

    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs, histogram_freq=1, profile_batch=(2, 5)
    )

    model.fit(
        train,
        epochs=epochs,
        shuffle=shuffle,
        validation_data=val,
        # callbacks=[tboard_callback],
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
    args = parser.parse_args()

    m = lstm_abstract_retriever(args.lstm_units)
    loss = tf.keras.losses.BinaryCrossentropy()
    m.compile(optimizer="adam", loss=loss)
    if args.train:
        train(m, batch_size=args.batch_size, epochs=args.epochs)
        save(m)
        m = load()
        m.summary()
