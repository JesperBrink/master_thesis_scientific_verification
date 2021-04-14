from models.sentence_selection.lstmModel import BertLSTMSentenceSelector
from tqdm import tqdm
import jsonlines
import argparse

def create_id_to_abstract_map(corpus_path):
    abstract_id_to_abstract = dict()
    corpus = jsonlines.open(corpus_path)
    for data in corpus:
        abstract_id_to_abstract[data["doc_id"]] = data["abstract"]
    return abstract_id_to_abstract

def get_gold_abstracts(claim_object, corpus):
    evidence = claim_object["evidence"]
    return {doc_id: corpus[int(doc_id)] for doc_id, _ in evidence.items()}

def evaluate(model, claim_path, corpus_path):
    corpus = create_id_to_abstract_map(corpus_path)
    with jsonlines.open(claim_path) as claims:
        correct_counter = 0
        wrong_counter = 0
        for claim_object in tqdm(claims):
            gold = get_gold_abstracts(claim_object, corpus)
            if gold == {}:
                continue
            selected_sentences = model(claim_object["claim"], gold)
            for doc_id, rationales in claim_object["evidence"].items():
                all_rationales = []
                for rationale in rationales:
                    all_rationales.extend(rationale["sentences"])
                selected_sentences[doc_id]
                correct = len([1 for x in selected_sentences if x in all_rationales])
                correct_counter += correct
                wrong_counter += (len(selected_sentences) - correct)
        print("accuracy", correct_counter / wrong_counter)

    pass

def main():
    parser = argparse.ArgumentParser(
        description="Script to run evaluation a sentence selection model"
    )
    parser.add_argument(
        "sentence_selector",
        metavar="sentence_selector",
        type=str,
        choices=["lstm"],
        help="Which sentence selection model to use. twolayer = Two layer dense, cosine = SBERT cosine similarity",
    )
    parser.add_argument(
        "claim_path", metavar="path", type=str, help="the path to the sentence claims"
    )
    parser.add_argument(
        "corpus_path",
        metavar="path",
        type=str,
        help="the path to the sentence corpus",
    )
    args = parser.parse_args()
    sentence_selector = None
    if args.sentence_selector == "lstm":
        sentence_selector = BertLSTMSentenceSelector(0.25)

    evaluate(sentence_selector, args.claim_path, args.corpus_path)
    pass

if __name__ == '__main__':
    main()
    