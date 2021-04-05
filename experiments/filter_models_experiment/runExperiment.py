from competition.pipeline import setup_sentence_embeddings
from models.filter_corpus.cosine_similarity import CosineSimilarityFilterModel
from models.filter_corpus.bm25 import BM25FilterModel
from utils.evaluationutils import compute_f1, compute_precision, compute_recall
import jsonlines
from tqdm import tqdm
import argparse
import enum
import time

def eval_sentence_selection(claims_path, corpus_path, level, model):
    sentence_embeddings, corp_id = setup_sentence_embeddings(corpus_path)
    measures = {}
    
    with jsonlines.open(claims_path) as claims:
        for claim in tqdm(claims):
            if not claim["evidence"]:
                continue

            k_range = [3,5, 10, 20, 30, 40, 50]
            top = model.get_top_k_by_similarity_with_ids(claim, sentence_embeddings, corp_id, k_range[-1], level)
            for k in k_range:
                if k not in measures:
                    measures[k] = {
                        "true_positives": 0, 
                        "false_positives": 0, 
                        "false_negatives": 0
                    }
                
                true_positives, false_positives, false_negatives = get_correct_at_top_k(claim, top, k, level)
                measures[k]["true_positives"] += true_positives
                measures[k]["false_positives"] += false_positives
                measures[k]["false_negatives"] += false_negatives
    
    for k in measures:
        precision = compute_precision(measures[k]["true_positives"], measures[k]["false_positives"])
        recall = compute_recall(measures[k]["true_positives"], measures[k]["false_negatives"])
        f1 = compute_f1(precision, recall)
        print("k: {}".format(k))
        print("precision: {}".format(precision))
        print("recall: {}".format(recall))
        print("f1: {}".format(f1))
        print("")

def get_correct_at_top_k(claim, top, k, level):
    top_k = top[:k]
    
    gold_docs = []
    for abstract_id, rationales in claim["evidence"].items():
        if level == "abstract":
            gold_docs.append(int(abstract_id))
        else:
            for rationale in rationales:
                for sentence_number in rationale["sentences"]:
                    gold_docs.append((int(abstract_id), sentence_number))

    retrieved = []
    for res in top_k:
        if hasattr(res, "numpy"):
            res = tuple(res.numpy())

        retrieved.append(res)

    true_positives = len(set(retrieved).intersection(set(gold_docs)))
    false_positives = len(retrieved) - true_positives
    false_negatives = len(gold_docs) - true_positives
    
    return true_positives, false_positives, false_negatives

class FilterModel(enum.Enum):
    SBERT_COSINE_SIMILARITY = "cosine"
    BM25 = "bm25"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracies of sentence selection")
    parser.add_argument(
        "claims", metavar="path", type=str, help="the path to the claims"
    )
    parser.add_argument(
        "corpus", metavar="path", type=str, help="the path to the corpus"
    )
    parser.add_argument(
        "level",
        type=str,
        choices=["sentence", "abstract"],
        help="Which level to use, abstract or sentence",
    )
    parser.add_argument(
        "filter_model",
        metavar="filter",
        type=FilterModel,
        help="Which pruning model to use. cosine = SBERT + cosine similarity, bm25 = BM25 Pyserini version (used in VERT5ERINI)",
    )
    args = parser.parse_args()

    if args.filter_model == FilterModel.SBERT_COSINE_SIMILARITY:
        filter_model = CosineSimilarityFilterModel()
    elif args.filter_model == FilterModel.BM25:
        filter_model = BM25FilterModel("../../datasets/scifact/corpus.jsonl", "bm25_corpus.jsonl", "bm25_index.jsonl", args.level)
    else:
        raise NotImplementedError()

    eval_sentence_selection(args.claims, args.corpus, args.level, filter_model)