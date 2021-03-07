import os
import argparse
import enum
import json 
import jsonlines
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import models.sentence_selection.model as sentence_selection_module
import models.stance_prediction.model as stance_prediction_module
from competition.pipeline import sentence_selection, setup_sentence_embeddings
from utils.evaluationutils import compute_f1, compute_precision, compute_recall


def keys_to_int(x):
    return {int(k): v for k, v in x.items()}


def load_hyperparameter_grid(path):
    with open(path) as json_file: 
        params = json.load(json_file) 
        return ParameterGrid(params)        


def evaluate_sentence_selection_model(model, embedded_validation_data_path, sentence_embeddings, corp_id, output_file):        
    total_abstract_true_positives = 0
    total_abstract_false_positives = 0
    total_abstract_false_negatives = 0
    
    total_sentence_true_positives = 0
    total_sentence_false_positives = 0
    total_sentence_false_negatives = 0
    
    with jsonlines.open(embedded_validation_data_path) as claims:
        for claim in tqdm(claims):
            relevant_sentences_dict = sentence_selection(claim, model, sentence_embeddings, corp_id)

            # Abstract retrieval
            retrieved_abstracts = relevant_sentences_dict.keys()
            gold_docs = [int(x) for x in claim["evidence"].keys()]
            
            abstract_true_positives = len(set(retrieved_abstracts).intersection(set(gold_docs)))
            abstract_false_positives = len(retrieved_abstracts) - abstract_true_positives
            abstract_false_negatives = len(gold_docs) - abstract_true_positives

            total_abstract_true_positives += abstract_true_positives
            total_abstract_false_positives += abstract_false_positives
            total_abstract_false_negatives += abstract_false_negatives

            # Sentence selection
            for abstract in retrieved_abstracts:                    
                if abstract not in gold_docs:
                    total_sentence_false_positives += len(relevant_sentences_dict[abstract])
                    continue

                predicted_sentences = [[x["id"]] for x in relevant_sentences_dict[abstract]]
                gold_rationales = [x["sentences"] for x in claim["evidence"][str(abstract)]]

                sentence_true_positives = len([1 for x in predicted_sentences if x in gold_rationales])
                sentence_false_positives = len(predicted_sentences) - sentence_true_positives
                sentence_false_negatives = len(gold_rationales) - sentence_true_positives

                total_sentence_true_positives += sentence_true_positives
                total_sentence_false_positives += sentence_false_positives
                total_sentence_false_negatives += sentence_false_negatives

        abstract_precision = compute_precision(total_abstract_true_positives, total_abstract_false_positives)
        abstract_recall = compute_recall(total_abstract_true_positives, total_abstract_false_negatives)
        abstract_f1 = compute_f1(abstract_precision, abstract_recall)

        output_file.write("Abstract Retrieval Precision: {}\n".format(abstract_precision))
        output_file.write("Abstract Retrieval Recall: {}\n".format(abstract_recall))
        output_file.write("Abstract Retrieval F1: {}\n".format(abstract_f1))

        sentence_precision = compute_precision(total_sentence_true_positives, total_sentence_false_positives)
        sentence_recall = compute_recall(total_sentence_true_positives, total_sentence_false_negatives)
        sentence_f1 = compute_f1(sentence_precision, sentence_recall)

        output_file.write("Sentence Selection Precision: {}\n".format(sentence_precision))
        output_file.write("Sentence Selection Recall: {}\n".format(sentence_recall))
        output_file.write("Sentence Selection F1: {}\n".format(sentence_f1))

       


def evaluate_hyperparameters_sentence_selection(train_data_path, validation_data_path, corpus_path):
    BATCH_SIZE = 32
    hyperparameter_grid = load_hyperparameter_grid("hyperparameter_dicts/sentence_level_hyperparameter_dict.json")
    sentence_embeddings, corp_id = setup_sentence_embeddings(corpus_path)

    if not os.path.exists("output"):
        os.makedirs("output")

    output_path = "output/{}-sentence-selection".format(datetime.now().strftime("%y%m%d%H%M%S"))
    with open(output_path, "w") as output_file:
        for hyper_parameters in hyperparameter_grid:				
            output_file.write("Params: {}\n".format(hyper_parameters))
            model = sentence_selection_module.initialize_model(BATCH_SIZE, hyper_parameters["dense_units"])
            class_weight = keys_to_int(hyper_parameters["class_weight"])
            model = sentence_selection_module.train(model, "scifact", BATCH_SIZE, class_weight)
            evaluate_sentence_selection_model(model, validation_data_path, sentence_embeddings, corp_id, output_file)
            output_file.write("#" * 50 + "\n")


def evaluate_hyperparameters_stance_prediction(train_data_path, validation_data_path):
    hyperparameters = load_hyperparameters("hyperparameter_dicts/sentence_level_hyperparameter_dict.json")
    raise NotADirectoryError 


class ProblemType(enum.Enum):
    SENTENCE_SELECTION = "sentence"
    STANCE_PREDICTION = "stance"


def main():
    parser = argparse.ArgumentParser(description="Script to evaluate different hyperparameters")
    parser.add_argument(
        "problem_type", metavar="problem_type", type=ProblemType, help="Type of problem to evaluate on"
    )
    parser.add_argument(
        "train_data_path", metavar="path", type=str, help="Path to tfrecords training file"
    )
    parser.add_argument(
        "validation_data_path", metavar="path", type=str, help="Path to jsonl validation file with embedded claims"
    )
    parser.add_argument(
        "corpus_path", metavar="path", type=str, help="Path to jsonl corpus file with embedded abstracts"
    )
    
    args = parser.parse_args()
    if args.problem_type == ProblemType.SENTENCE_SELECTION:
        evaluate_hyperparameters_sentence_selection(args.train_data_path, args.validation_data_path, args.corpus_path)
    elif args.problem_type == ProblemType.STANCE_PREDICTION:
        evaluate_hyperparameters_stance_prediction(args.train_data_path, args.validation_data_path)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
    