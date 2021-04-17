import argparse
import jsonlines
import tqdm

from competition.pipeline import sentence_selection, setup_sentence_embeddings
from models.sentence_selection.cosine_similarity_model import CosineSimilaritySentenceSelector


def main(embedded_dev_claim_path, embedded_corpus_path, threshold):
    # for each claim in dev claims
        # compute sentences above threshold (like in the pipeline)
        # compare to sentences in ground truth
        # print if they get correct or wrong abstract (for each abstract)
        # print if they get correct or wrong sentences (for each sentence)
    
    sentence_selection_model = CosineSimilaritySentenceSelector()
    sentence_embeddings, corp_id = setup_sentence_embeddings(embedded_corpus_path)
    
    with jsonlines.open(embedded_dev_claim_path) as claims:
        for claim in tqdm(claims):
            relevant_sentences_dict = sentence_selection(claim, sentence_selection_model, sentence_embeddings, corp_id, threshold)
            print("Relevant sentence dict", relevant_sentences_dict)
            exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyse the performance of our SBERT-based model on Rationale Selection"
    )
    parser.add_argument(
        "embedded_dev_claim_path",
        help="the path to the jsonl file with embedded claims",
    )
    parser.add_argument(
        "embedded_corpus_path",
        help="path to the jsonl file with embedded corpora",
    )
    parser.add_argument(
        "threshold",
        help="threshold that should be used for the sentence selection model",
        type=float,
    )

    args = parser.parse_args()

    main(args.embedded_dev_claim_path, args.embedded_corpus_path, args.threshold)

