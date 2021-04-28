import os
import argparse

from sentence_transformers import SentenceTransformer, losses, models
from torch.utils.data import DataLoader
from torch import nn

from load_data import load_training_data, load_evaluator


def initialize_model(pretrained_model, with_dense_layer=False):
    if not with_dense_layer:
        return SentenceTransformer(pretrained_model)

    word_embedding_model = models.Transformer(pretrained_model)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
    
    return SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])


def finetune_sbert(pretrained_model, data_folder_path, fever_epochs, scifact_epochs, with_dense_layer):
    scifact_training_data, fever_training_data = load_training_data(data_folder_path)
    scifact_train_dataloader = DataLoader(scifact_training_data, shuffle=True, batch_size=32)
    fever_train_dataloader = DataLoader(fever_training_data, shuffle=True, batch_size=32)
    evaluator = load_evaluator(data_folder_path)

    if fever_epochs > 0:
        fever_model = initialize_model(pretrained_model, with_dense_layer)
        train_loss = losses.CosineSimilarityLoss(fever_model)
        fever_output_name = get_output_name(pretrained_model, "fever", fever_epochs, with_dense_layer)

        if  os.path.exists(fever_output_name):
            print("Adding 'NEW' to model name, instead of overwriting existing")
            fever_output_name += "-NEW"

        fever_model.fit(
            train_objectives=[(fever_train_dataloader, train_loss)],
            epochs=fever_epochs,
            evaluator=evaluator,
            evaluation_steps=500,
            output_path=fever_output_name,
        )

    if scifact_epochs > 0:
        if fever_epochs > 0:
            scifact_model = SentenceTransformer(fever_output_name)
            scifact_output_name = get_output_name(pretrained_model, "fever-{}-scifact".format(fever_epochs), scifact_epochs, with_dense_layer)
        else:
            scifact_model = initialize_model(pretrained_model, with_dense_layer)
            scifact_output_name = get_output_name(pretrained_model, "scifact", scifact_epochs, with_dense_layer)

        scifact_train_loss = losses.CosineSimilarityLoss(scifact_model)

        if  os.path.exists(scifact_output_name):
            print("Adding 'NEW' to model name, instead of overwriting existing")
            scifact_output_name += "-NEW"

        scifact_model.fit(
            train_objectives=[(scifact_train_dataloader, scifact_train_loss)],
            epochs=scifact_epochs,
            evaluator=evaluator,
            evaluation_steps=500,
            output_path=scifact_output_name,
        )


def get_output_name(pretrained_model, dataset_type, epochs, with_dense_layer):
    output = "{}-finetuned-on-{}-{}".format(pretrained_model, dataset_type, epochs)

    if with_dense_layer:
        output += "-dense"
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a claims")
    parser.add_argument(
        "pretrained_model",
        metavar="path",
        type=str,
        help="the name of (or path to, if local) the pretrained sbert model",
    )
    parser.add_argument(
        "data_folder_path",
        metavar="path",
        type=str,
        help="path to folder with result from running create_data.py",
    )
    parser.add_argument(
        "fever_epochs",
        help="numbber of epochs on fever (before finetuning on scifact)",
        type=int,
    )
    parser.add_argument(
        "scifact_epochs",
        help="number of epochs on scifact",
        type=int,
    )
    parser.add_argument(
        "-d",
        "--dense",
        help="set if you want the model to have a dense layer",
        action="store_true",
    ) 

    args = parser.parse_args()
    finetune_sbert(args.pretrained_model, args.data_folder_path, args.fever_epochs, args.scifact_epochs, args.dense)
