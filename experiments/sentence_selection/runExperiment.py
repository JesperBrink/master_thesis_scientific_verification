from competition.pipeline import setup_sentence_embeddings
from models.filter_corpus.cosine_similarity import CosineSimilarityFilterModel
import jsonlines
from tqdm import tqdm
from collections import defaultdict
import argparse

def eval_sentence_selection(claims_path, corpus_path, model):
    sentence_embeddings, corp_id = setup_sentence_embeddings(corpus_path)
    accuracies = defaultdict(list)
    
    with jsonlines.open(claims_path) as claims:
        for claim in tqdm(claims):
            if not claim["evidence"]:
                continue

            top_50 = model.get_top_k_by_similarity_with_ids(claim, sentence_embeddings, corp_id, 50)
            for k in [10,20,30,40,50]:
                accuracies[k].append(get_correct_at_top_k(claim, top_50, k))

    for k in accuracies:
        avg = sum(accuracies[k]) / len(accuracies[k])
        print("avg. acc. at k={}: {}".format(k, avg))
    

def get_correct_at_top_k(claim, top_50, k):
    top_k = top_50[:k]
    correct = 0
    total = 0


    for rationales in claim["evidence"].values():
        for rationale in rationales:
            total += len(rationale["sentences"])

    for res in top_k:
        abstract_id, sentence_number = res.numpy()
        if str(abstract_id) not in claim["evidence"]: 
            continue
        for rationale in claim["evidence"][str(abstract_id)]:
            if sentence_number in rationale["sentences"]:
                correct += 1
    
    return correct/total
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracies of sentence selection")
    parser.add_argument(
        "claims", metavar="path", type=str, help="the path to the claims"
    )
    parser.add_argument(
        "corpus", metavar="path", type=str, help="the path to the corpus"
    )
    args = parser.parse_args()

    model = CosineSimilarityFilterModel()

    eval_sentence_selection(args.claims, args.corpus, model)