from competition.pipeline import setup_sentence_embeddings
from models.filter_corpus.bm25 import BM25FilterModel
from models.filter_corpus.oracle import OracleFilterModel
from models.filter_corpus.hybridv3 import HybridV3FilterModel
from models.filter_corpus.hybridv2 import HybridV2FilterModel
from models.filter_corpus.hybrid import HybridFilterModel
from models.filter_corpus.cosine_similarity import CosineSimilarityFilterModel
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
            #min_k = 5183 #min(len(sentence_embeddings), 5183)
            k_range = [3, 10, 100, 150]
            #top = model.get_top_k_by_similarity_with_ids(claim, sentence_embeddings, corp_id, min_k, level)
            top = model.get_top_k_by_similarity_with_ids(claim, sentence_embeddings, corp_id, k_range[-1], level)
            #for k in range(1, min_k+1):
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
        recall = compute_recall(measures[k]["true_positives"], measures[k]["false_negatives"])
        precision = compute_precision(measures[k]["true_positives"], measures[k]["false_positives"])
        f1 = compute_f1(precision, recall)
        print(f"k: {k}, recall: {recall}, precision: {precision}, f1: {f1}")
        

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

    retrieved = top_k
    true_positives = len(set(retrieved).intersection(set(gold_docs)))
    false_positives = len(retrieved) - true_positives
    false_negatives = len(gold_docs) - true_positives
    
    return true_positives, false_positives, false_negatives

class FilterModel(enum.Enum):
    SBERT_COSINE_SIMILARITY = "cosine"
    BM25 = "bm25"
    ORACLE = "oracle"
    SPECTER = "specter"
    SPECTERAPI = "specterAPI"
    HYBRID = "hybrid"
    HYBRIDV2 = "hybridv2"
    HYBRIDV3 = "hybridv3"

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
        filter_model = BM25FilterModel(args.corpus, "bm25_corpus.jsonl", "bm25_index.jsonl", args.level)
    elif args.filter_model == FilterModel.ORACLE:
        filter_model = OracleFilterModel()
    elif args.filter_model == FilterModel.SPECTER:
        from models.filter_corpus.specter import SpecterFilterModel
        filter_model = SpecterFilterModel(args.corpus)
    elif args.filter_model == FilterModel.SPECTERAPI:
        from models.filter_corpus.specterAPI import SpecterAPIFilterModel
        filter_model = SpecterAPIFilterModel(args.corpus)
    elif args.filter_model == FilterModel.HYBRID:
        filter_model = HybridFilterModel(args.corpus, "bm25_corpus.jsonl", "bm25_index.jsonl", args.level)
    elif args.filter_model == FilterModel.HYBRIDV2:
        filter_model = HybridV2FilterModel(args.corpus, "bm25_corpus.jsonl", "bm25_index.jsonl", args.level)
    elif args.filter_model == FilterModel.HYBRIDV3:
        filter_model = HybridV3FilterModel(args.corpus, "bm25_corpus.jsonl", "bm25_index.jsonl")
    else:
        raise NotImplementedError()

    eval_sentence_selection(args.claims, args.corpus, args.level, filter_model)