import tensorflow as tf
import os
import pandas as pd
from csv import reader
from pathlib import Path
import numpy as np


dataset_path = Path(os.path.realpath(__file__)).resolve().parents[1] / "trainingDataset"

def load_dataset():
    df_relevant = pd.read_csv(dataset_path / "relevant.csv", header=None)
    relevant_features = np.array(df_relevant.iloc[:,0])
    relevant_labels = np.ones_like(relevant_features)
    print(relevant_features, relevant_labels)

    df_not_relevant = pd.read_csv(dataset_path / "not_relevant.csv", header=None)
    not_relevant_features = np.array(df_not_relevant.iloc[:,0])
    not_relevant_labels = np.zeros_like(not_relevant_features)

    #for e in dataset.take():
    #    print(e)


if __name__ == '__main__':
    load_dataset()
    