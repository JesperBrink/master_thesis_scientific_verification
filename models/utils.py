import os
import shutil
import datetime
import tensorflow as tf


def get_highest_count(dir):
    m = -1
    for file in os.listdir(dir):
        number = int(file.split('_')[-1])
        if number > m: 
            m = number
    return m


def setup_tensorboard():
    # clear old logs
    if os.path.exists("./logs"):
        shutil.rmtree("./logs")

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
