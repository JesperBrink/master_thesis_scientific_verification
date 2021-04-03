import os
import argparse

from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from load_data import load_training_data, load_evaluator


def finetune_sbert(pretrained_model, fever_epochs, scifact_epochs, preprocess_stopwords):
    scifact_training_data, fever_training_data = load_training_data(preprocess_stopwords)
    scifact_train_dataloader = DataLoader(scifact_training_data, shuffle=True, batch_size=32)
    fever_train_dataloader = DataLoader(fever_training_data, shuffle=True, batch_size=32)
    evaluator = load_evaluator(preprocess_stopwords)

    if fever_epochs > 0:
        fever_model = SentenceTransformer(pretrained_model)
        train_loss = losses.CosineSimilarityLoss(fever_model)
        fever_output_name = get_output_name(pretrained_model, "fever", fever_epochs, preprocess_stopwords)

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
            scifact_output_name = get_output_name(pretrained_model, "fever-{}-scifact".format(fever_epochs), scifact_epochs, preprocess_stopwords)
        else:
            scifact_model = SentenceTransformer(pretrained_model)
            scifact_output_name = get_output_name(pretrained_model, "scifact", scifact_epochs, preprocess_stopwords)

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


def get_output_name(pretrained_model, dataset_type, epochs, preprocess_stopwords):
    if preprocess_stopwords:
        return "{}-finetuned-on-{}-{}-no-stopwords".format(pretrained_model, dataset_type, epochs)
    
    return "{}-finetuned-on-{}-{}".format(pretrained_model, dataset_type, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a claims")
    parser.add_argument(
        "pretrained_model",
        metavar="path",
        type=str,
        help="the name of (or path to, if local) the pretrained sbert model",
    )   
    parser.add_argument(
        "-f",
        "--fever_epochs",
        help="numbber of epochs on fever (before finetuning on scifact)",
        type=int,
    )
    parser.add_argument(
        "-s",
        "--scifact_epochs",
        help="number of epochs on scifact",
        type=int,
    ) 
    parser.add_argument(
        "-p",
        "--preprocess_stopwords",
        help="set if you want to finetune on data where stopwords have been removed",
        action="store_true",
    )

    args = parser.parse_args()
    finetune_sbert(args.pretrained_model, args.fever_epochs, args.scifact_epochs, args.preprocess_stopwords)
    