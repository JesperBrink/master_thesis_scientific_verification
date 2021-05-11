from competition.pipeline import setup_sentence_embeddings
from models.filter_corpus.bm25 import BM25FilterModel
from models.filter_corpus.hybridv2 import HybridV2FilterModel
from models.filter_corpus.hybrid import HybridFilterModel
from utils.evaluationutils import compute_f1, compute_precision, compute_recall
import jsonlines
from tqdm import tqdm
import argparse
import enum
import numpy as np
import os

def eval_sentence_selection(claims_path, corpus_path, model, k1, b, res_file):
    sentence_embeddings, corp_id = setup_sentence_embeddings(corpus_path)
    measures = {}
    
    with jsonlines.open(claims_path) as claims:
        for claim in claims:
            if not claim["evidence"]:
                continue

            # Get top 100 (just to reduce search space) and calculate recall for all k 1-100
            top = model.get_top_k_by_similarity_with_ids(claim, sentence_embeddings, corp_id, 100, "haha")
            for k in range(1,101):
                if k not in measures:
                    measures[k] = {
                        "true_positives": 0, 
                        "false_positives": 0, 
                        "false_negatives": 0
                    }
                
                true_positives, false_positives, false_negatives = get_correct_at_top_k(claim, top, k)
                measures[k]["true_positives"] += true_positives
                measures[k]["false_positives"] += false_positives
                measures[k]["false_negatives"] += false_negatives
    
        with open(res_file, 'a') as f:
            f.write("k1: {}, b: {}\n".format(k1,b))
            found = False
            for k in measures:
                if found:
                    continue
                recall = compute_recall(measures[k]["true_positives"], measures[k]["false_negatives"])
                if recall >= 0.95:
                    f.write("{}, {}\n".format(k, recall))
                    found = True

def get_correct_at_top_k(claim, top, k):
    top_k = top[:k]
    
    gold_docs = []
    for abstract_id, rationales in claim["evidence"].items():
            gold_docs.append(int(abstract_id))

    retrieved = []
    for res in top_k:
        if hasattr(res, "numpy"):
            res = tuple(res.numpy())

        retrieved.append(res)

    true_positives = len(set(retrieved).intersection(set(gold_docs)))
    false_positives = len(retrieved) - true_positives
    false_negatives = len(gold_docs) - true_positives
    
    return true_positives, false_positives, false_negatives


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracies of sentence selection")
    parser.add_argument(
        "claims", metavar="path", type=str, help="the path to the claims"
    )
    parser.add_argument(
        "corpus", metavar="path", type=str, help="the path to the corpus"
    )
    args = parser.parse_args()

    res_file = "res_paren.txt"

    if os.path.exists(res_file):
       os.remove(res_file)
    
    for k1 in tqdm(np.arange(1,2.1, 0.1)):
       for b in np.arange(0,1.1,0.1):
           k1 = round(k1,2)
           b = round(b,2)

           filter_model = BM25FilterModel(args.corpus, "bm25_corpus.jsonl", "bm25_index.jsonl", "abstract", k1, b)
           eval_sentence_selection(args.claims, args.corpus, filter_model, k1, b, res_file)
    
    # find best hyperparams
    with open(res_file, 'r') as f:
        best_no = 10000
        best_recall = 0
        best_k1 = 0
        best_b = 0
        for line in f:
            if "k1" in line:
                e = line.split()
                k1 = float(e[1][:-1])
                b = float(e[3])
                continue
            l = line.split(",")
            if int(l[0]) < best_no or (int(l[0]) == best_no and float(l[1][:-2]) >= best_recall):
                best_no = int(l[0])
                best_recall = float(l[1][:-2])
                best_k1 = k1
                best_b = b
        print(best_no, best_recall, best_k1, best_b)

