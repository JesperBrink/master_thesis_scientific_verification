import os
import argparse
import enum
import json
import jsonlines
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from models.sentence_selection import denseModel as sentence_denseModel
from models.stance_prediction import denseModel as stance_denseModel
from utils.evaluationutils import compute_f1, compute_precision, compute_recall
from external_scripts.eval import run_evaluation


def create_id_to_abstract_map(corpus_path):
    abstract_id_to_abstract = dict()
    corpus = jsonlines.open(corpus_path)
    for data in corpus:
        abstract_id_to_abstract[data["doc_id"]] = data["abstract"]

    return abstract_id_to_abstract


def load_hyperparameter_grid(path):
    with open(path) as json_file:
        params = json.load(json_file)
        thresholds = params.pop("threshold", None)
        return ParameterGrid(params), thresholds


def get_abstract_id_to_abstract_embedding_map(corpus_path):
    abstract_id_to_abstract_embedding_map = dict()
    with jsonlines.open(corpus_path) as corpus_reader:
        for line in corpus_reader:
            abstract_id_to_abstract_embedding_map[line["doc_id"]] = line["abstract"]

    return abstract_id_to_abstract_embedding_map


def convert_to_scifact_format(claim_id, pred):
    if not pred:
        return {"id": claim_id, "evidence": {}}

    evidence = dict()
    for abstract in pred.keys():
        evidence[str(abstract)] = {
            "sentences": pred[abstract],
            "label": "SUPPORT",
        }  # "SUPPORT" is just a dummy variable

    return {"id": claim_id, "evidence": evidence}


def make_evidence_from_grund_truth(claim, abstract_id_to_abstract_embedding_map):
    evidence = dict()
    for abstract, sentence_list in claim["evidence"].items():
        sentence_id_list = []

        for sent_dict in sentence_list:
            for sent_id in sent_dict["sentences"]:
                sentence_id_list.append(sent_id)
        evidence[abstract] = sentence_id_list

    return evidence


def evaluate_stance_predicion_model(
    model, claims_path, abstract_id_to_abstract_embedding_map
):
    predictions_list = []

    with jsonlines.open(claims_path) as claims:
        for claim in claims:
            evidence = make_evidence_from_grund_truth(
                claim, abstract_id_to_abstract_embedding_map
            )
            prediction = model(claim, evidence, None)
            predictions_list.append(prediction)

    labels_file = "../datasets/scifact/claims_validation.jsonl"
    metrics = run_evaluation(labels_file, predictions_list)

    results_dict = {
        "Abstract Label Only Precision": metrics["abstract_label_only_precision"],
        "Abstract Label Only Recall": metrics["abstract_label_only_recall"],
        "Abstract Label Only F1": metrics["abstract_label_only_f1"],
        "Sentence Label Precision": metrics["sentence_label_precision"],
        "Sentence Lable Recall": metrics["sentence_label_recall"],
        "Sentence Label F1": metrics["sentence_label_f1"],
    }

    return results_dict


def evaluate_sentence_selection_model(
    model, 
    abstracts,
    claims_embedding_path,
):
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    predictions_list = []

    # Abstract retrieval
    with jsonlines.open(claims_embedding_path) as claims:
        for claim in tqdm(claims):
            relevant_sentences_dict = model(claim, abstracts)
            
            predictions_list.append(
                convert_to_scifact_format(claim["id"], relevant_sentences_dict)
            )

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

    labels_file = "../datasets/scifact/claims_validation.jsonl"
    metrics = run_evaluation(labels_file, predictions_list)

    result_dict = {
        "Abstract Retrieval Precision": precision,
        "Abstract Retrieval Recall": recall,
        "Abstract Retrieval F1": f1,
        "Sentence Selection Precision": metrics["sentence_selection_precision"],
        "Sentence Selection Recall": metrics["sentence_selection_recall"],
        "Sentence Selection F1": metrics["sentence_selection_f1"],
    }

    return result_dict


def evaluate_hyperparameters_sentence_selection(claims_path, corpus_path):
    BATCH_SIZE = 32
    hyperparameter_grid, thresholds = load_hyperparameter_grid(
        "hyperparameter_dicts/sentence_selection_hyperparameter_dict.json"
    )
    
    abstracts = create_id_to_abstract_map(corpus_path)
    
    if not os.path.exists("output"):
        os.makedirs("output")

    output_path = "output/{}-sentence-selection".format(
        datetime.now().strftime("%y%m%d%H%M%S")
    )
    with jsonlines.open(output_path, "w", flush=True) as output_file:
        for hyper_parameters in hyperparameter_grid:
            model = sentence_denseModel.setup_for_training(
                hyper_parameters["dense_units"],
                hyper_parameters["learning_rate"],
            )
            class_weight = {
                0: int(hyper_parameters["class_weight_0"]),
                1: int(hyper_parameters["class_weight_1"]),
            }

            if class_weight[0] == class_weight[1] and class_weight[0] > 1:
                continue

            model = sentence_denseModel.train(
                model,
                "fever",
                BATCH_SIZE,
                hyper_parameters["fever_epochs"],
                class_weight,
            )
            model = sentence_denseModel.train(
                model,
                "scifact",
                BATCH_SIZE,
                hyper_parameters["scifact_epochs"],
                class_weight,
            )

            prediction_model = sentence_denseModel.TwoLayerDenseSentenceSelector(
                corpus_path,
                claims_path,
                model=model
            )

            for threshold in thresholds:
                prediction_model.threshold = threshold
                result_dict = evaluate_sentence_selection_model(
                    prediction_model,
                    abstracts,
                    claims_path
                )
                hyper_parameters["threshold"] = threshold
                output_file.write({"params": hyper_parameters, "results": result_dict})


def evaluate_hyperparameters_stance_prediction(claims_path, corpus_path):
    BATCH_SIZE = 32
    hyperparameter_grid, _ = load_hyperparameter_grid(
        "hyperparameter_dicts/stance_prediction_hyperparameter_dict.json"
    )
    abstract_id_to_abstract_embedding_map = get_abstract_id_to_abstract_embedding_map(
        corpus_path
    )

    if not os.path.exists("output"):
        os.makedirs("output")

    output_path = "output/{}-stance-prediction".format(
        datetime.now().strftime("%y%m%d%H%M%S")
    )
    with jsonlines.open(output_path, "w", flush=True) as output_file:
        for hyper_parameters in hyperparameter_grid:
            model = stance_denseModel.setup_for_training(
                hyper_parameters["dense_units"],
                hyper_parameters["learning_rate"],
            )
            model = stance_denseModel.train(
                model, "fever", BATCH_SIZE, hyper_parameters["fever_epochs"]
            )
            model = stance_denseModel.train(
                model, "scifact", BATCH_SIZE, hyper_parameters["scifact_epochs"]
            )

            prediction_model = stance_denseModel.TwoLayerDenseStancePredictor(
                corpus_path,
                claims_path,
                model=model
            )

            result_dict = evaluate_stance_predicion_model(
                prediction_model, claims_path, abstract_id_to_abstract_embedding_map
            )
            output_file.write({"params": hyper_parameters, "results": result_dict})


class ProblemType(enum.Enum):
    SENTENCE_SELECTION = "sentence"
    STANCE_PREDICTION = "stance"


def main():
    parser = argparse.ArgumentParser(
        description="Script to evaluate different hyperparameters"
    )
    parser.add_argument(
        "problem_type",
        metavar="problem_type",
        type=ProblemType,
        help="Type of problem to evaluate on",
    )
    parser.add_argument(
        "claims_path",
        metavar="path",
        type=str,
        help="Path to jsonl validation file with embedded claims",
    )
    parser.add_argument(
        "corpus_path",
        metavar="path",
        type=str,
        help="Path to jsonl corpus file with embedded abstracts",
    )

    args = parser.parse_args()
    if args.problem_type == ProblemType.SENTENCE_SELECTION:
        evaluate_hyperparameters_sentence_selection(args.claims_path, args.corpus_path)
    elif args.problem_type == ProblemType.STANCE_PREDICTION:
        evaluate_hyperparameters_stance_prediction(args.claims_path, args.corpus_path)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
