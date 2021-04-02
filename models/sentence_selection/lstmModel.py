import os
from pathlib import Path

import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertConfig

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
    config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False

    inputs = tf.keras.Input(shape=(3,128,), dtype="int32", name="sequence")
    inputs_mask = tf.keras.Input(
        shape=(3, 128,), dtype="int32", name="attention_masks"
    )
    reshaped_inputs = tf.reshape(inputs,[-1,128], name="seq_reshape")
    reshaped_mask = tf.reshape(inputs_mask,[-1,128], name="seq_mask_reshape")

    embedding = TFDistilBertModel.from_pretrained(
        "distilbert-base-uncased", config=config
    )(reshaped_inputs, attention_mask=reshaped_mask)[0]
    cls_tokens = embedding[:,0,:]
    reshaped_cls = tf.reshape(cls_tokens, [-1, 3, 768], name="cls_reshape")

    lstm = tf.keras.layers.LSTM(
        untis, return_sequences=False, recurrent_initializer="glorot_uniform"
    )(reshaped_cls)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(lstm)

    model = tf.keras.Model(
        inputs=[inputs, inputs_mask], outputs=outputs, name="bert-lstm-abstract-retriever"
    )

    model.summary()

    return model

def train(model, epochs=10, batch_size=16, shuffle=True):
    train = ScifactLSTMDataset(DatasetType.train).load().batch(batch_size)
    val = ScifactLSTMDataset(DatasetType.validation).load().batch(batch_size)

    model.fit(
        train,
        epochs=epochs,
        shuffle=shuffle,
        validation_data=val,
    )

# TODO: doesn't work right so find other way of doing thi.
def load():
    count = get_highest_count(_model_dir)
    path = str(_model_dir / "bert_lstm_abstract_retriver_{}".format(count))
    model = tf.keras.models.load_model(path)
    return model

# TODO: doesn't work right
def save(model):
    count = get_highest_count(_model_dir) + 1
    path = str(_model_dir / "bert_lstm_abstract_retriver_{}".format(count))
    model.save(path)
    print("model saved to {}".format(path))

if __name__ == "__main__":
    m = lstm_abstract_retriever(512)
    loss = tf.keras.losses.BinaryCrossentropy()
    m.compile(optimizer="adam", loss=loss)
    
    save(m)
    loaded = load()
    loaded.summary()

