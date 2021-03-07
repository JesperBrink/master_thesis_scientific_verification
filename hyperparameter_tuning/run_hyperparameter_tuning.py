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
from external_scripts.eval import run_evaluation

# TODO: Vi skal tage 10% ud af træningsdataen til at lave et validation set 
    # nuværende validation set skal renames til Dev
    # det nye train sæt der består af 90% af train kan vi kalde sub_train
# TODO: brug deres script til at udregne sentence selection (og derefter også stance prediction)
# TODO: stance prediction
# TODO: jeg gør ikke brug af threshold parameteren
# TODO: kun print det relevante entries i sentence selection og stance prediction


def keys_to_int(x):
    return {int(k): v for k, v in x.items()}


def load_hyperparameter_grid(path):
    with open(path) as json_file: 
        params = json.load(json_file) 
        return ParameterGrid(params)        


def convert_to_scifact_format(predictions_list):
    result = []
    for claim_id, pred in predictions_list:
        if not pred:
            result.append({"id": claim_id, "evidence": {}})
            continue

        evidence = dict()
        for abstract in pred.keys():
            predicted_sentences = [sentence_dict["id"] for sentence_dict in pred[abstract]]
            evidence[str(abstract)] = {"sentences": predicted_sentences, "label": "SUPPORT"} # The zero is just a dummy variable

        result.append({"id": claim_id, "evidence": evidence})
    
    return result


def evaluate_sentence_selection_model(model, embedded_validation_data_path, sentence_embeddings, corp_id, output_file):        
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    predictions_list = []

    # Abstract retrieval
    with jsonlines.open(embedded_validation_data_path) as claims:
        for claim in tqdm(claims):
            relevant_sentences_dict = sentence_selection(claim, model, sentence_embeddings, corp_id)
            predictions_list.append((claim["id"], relevant_sentences_dict))

            retrieved_abstracts = relevant_sentences_dict.keys()
            gold_docs = [int(x) for x in claim["evidence"].keys()]
            
            true_positives = len(set(retrieved_abstracts).intersection(set(gold_docs)))
            false_positives = len(retrieved_abstracts) - true_positives
            false_negatives = len(gold_docs) - true_positives

            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives

        precision = compute_precision(total_true_positives, total_false_positives)
        recall = compute_recall(total_true_positives, total_false_negatives)
        f1 = compute_f1(precision, recall)

        output_file.write("Abstract Retrieval Precision: {}\n".format(precision))
        output_file.write("Abstract Retrieval Recall: {}\n".format(recall))
        output_file.write("Abstract Retrieval F1: {}\n".format(f1))

    # Scifact evaluation measures
    labels_file = "../datasets/scifact/claims_dev.jsonl" # TODO: what should this actually be?
    predictions_list = convert_to_scifact_format(predictions_list)
    metrics = run_evaluation(labels_file, predictions_list)
    json.dump(metrics, output_file, indent=4)


def evaluate_hyperparameters_sentence_selection(train_data_path, validation_data_path, corpus_path):
    BATCH_SIZE = 32
    hyperparameter_grid = load_hyperparameter_grid("hyperparameter_dicts/sentence_level_hyperparameter_dict.json")
    sentence_embeddings, corp_id = setup_sentence_embeddings(corpus_path)

    if not os.path.exists("output"):
        os.makedirs("output")

    output_path = "output/{}-sentence-selection".format(datetime.now().strftime("%y%m%d%H%M%S"))
    with open(output_path, "w", buffering=1) as output_file:
        for hyper_parameters in hyperparameter_grid:				
            output_file.write("Params: {}\n".format(hyper_parameters))
            model = sentence_selection_module.initialize_model(BATCH_SIZE, hyper_parameters["dense_units"])
            class_weight = keys_to_int(hyper_parameters["class_weight"])
            model = sentence_selection_module.train(model, "fever", BATCH_SIZE, class_weight)
            model = sentence_selection_module.train(model, "scifact", BATCH_SIZE, class_weight)
            evaluate_sentence_selection_model(model, validation_data_path, sentence_embeddings, corp_id, output_file)
            output_file.write("\n" + "#" * 50 + "\n")


def evaluate_hyperparameters_stance_prediction(train_data_path, validation_data_path):
    hyperparameters = load_hyperparameters("hyperparameter_dicts/sentence_level_hyperparameter_dict.json")
    raise NotImplementedError 


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
    