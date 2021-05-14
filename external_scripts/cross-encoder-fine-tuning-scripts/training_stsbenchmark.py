"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continious labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)
#### /print debug information to stdout


# Check if dataset exsist. If not, download and extract  it
sts_dataset_path = "datasets/stsbenchmark.tsv.gz"

if not os.path.exists(sts_dataset_path):
    util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)


# Define our Cross-Encoder
train_batch_size = 2
num_epochs = 4
model_save_path = (
    "output/cross-encoder-stsb-nli-allenai-scibert-scivocab-uncased-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

#############################################################################################################
# Needs to replace the classification layer, to replace it with a layer with a single output instead of three
model = CrossEncoder(
    "output/cross-encoder-nli-allenai-scibert-scivocab-uncased", num_labels=3
)
model.model.classifier = nn.Linear(768, 1, bias=True)
model.model.num_labels = 1
model.config.num_labels = 1

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.model = model.model.to(model._target_device)
# print(model.model.num_labels)
#############################################################################################################

# Read STSb dataset
logger.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1

        if row["split"] == "dev":
            dev_samples.append(
                InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)
            )
        elif row["split"] == "test":
            test_samples.append(
                InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)
            )
        else:
            # As we want to get symmetric scores, i.e. CrossEncoder(A,B) = CrossEncoder(B,A), we pass both combinations to the train set
            train_samples.append(
                InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)
            )
            train_samples.append(
                InputExample(texts=[row["sentence2"], row["sentence1"]], label=score)
            )


# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


# We add an evaluator, which evaluates the performance during training
evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name="sts-dev")


# Configure the training
warmup_steps = math.ceil(
    len(train_dataloader) * num_epochs * 0.1
)  # 10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
)


##### Load model and eval on test set
model = CrossEncoder(model_save_path)

evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name="sts-test")
evaluator(model)