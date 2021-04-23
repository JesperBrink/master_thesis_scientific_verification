from tqdm import tqdm
import jsonlines
from random import sample
from scipy import spatial
import time
import os
import pickle
import argparse

from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample


def create_training_data(scifact_corpus_path, scifact_train_path, fever_claims_path, embedding_model, output_path):
    scifact_training_data = []

    scifact_relevant_data = create_scifact_relevant(scifact_train_path, scifact_corpus_path)
    scifact_relevant_data_train_format = convert_to_train_format(scifact_relevant_data)
    scifact_training_data.extend(scifact_relevant_data_train_format)

    scifact_not_relevant_data = create_scifact_not_relevant(scifact_train_path, scifact_corpus_path, 5, embedding_model)
    scifact_not_relevant_data_train_format = convert_to_train_format(scifact_not_relevant_data)
    scifact_training_data.extend(scifact_not_relevant_data_train_format)

    fever_data = create_fever(fever_claims_path)
    fever_data_train_format = convert_to_train_format(fever_data)

    with open(os.path.join(output_path, "scifact_training_data.p"), "wb") as fp:
        pickle.dump(scifact_training_data, fp)

    with open(os.path.join(output_path, "fever_training_data.p"), "wb") as fp:
        pickle.dump(fever_data_train_format, fp)


def create_evaluator_data(scifact_corpus_path, scifact_validation_path, embedding_model, output_path):
    claims = []
    sentences = []
    labels = []

    scifact_relevant_data = create_scifact_relevant(scifact_validation_path, scifact_corpus_path)
    claims_relevant, sentences_relevant, labels_relevant = convert_to_evaluator_format(scifact_relevant_data)
    claims.extend(claims_relevant)
    sentences.extend(sentences_relevant)
    labels.extend(labels_relevant)

    scifact_not_relevant_data = create_scifact_not_relevant(scifact_validation_path, scifact_corpus_path, 5, embedding_model)
    claims_not_relevant, sentences_not_relevant, labels_not_relevant = convert_to_evaluator_format(scifact_not_relevant_data)
    claims.extend(claims_not_relevant)
    sentences.extend(sentences_not_relevant)
    labels.extend(labels_not_relevant)

    with open(os.path.join(output_path, "claims.p"), "wb") as fp:
        pickle.dump(claims, fp)

    with open(os.path.join(output_path, "sentences.p"), "wb") as fp:
        pickle.dump(sentences, fp)

    with open(os.path.join(output_path, "labels.p"), "wb") as fp:
        pickle.dump(labels, fp)


def convert_to_train_format(data):
    return [InputExample(texts=[claim, sentence], label=label) for claim, sentence, label in data]


def convert_to_evaluator_format(data):
    claims = [tup[0] for tup in data]
    sentences = [tup[1] for tup in data]
    labels = [tup[2] for tup in data]

    return claims, sentences, labels


def create_id_to_abstract_map(corpus_path):
    abstract_id_to_abstract = dict()
    corpus = jsonlines.open(corpus_path)
    for data in corpus:
        abstract_id_to_abstract[str(data["doc_id"])] = data["abstract"]

    return abstract_id_to_abstract


def create_scifact_relevant(claims_path, corpus_path):
    relevant_data = []

    id_to_abstact_map = create_id_to_abstract_map(corpus_path)
    
    for claim in tqdm(jsonlines.open(claims_path)):
        if not claim["evidence"]:
            continue

        sentences = []
        
        for doc_id, evidence_sets in claim["evidence"].items():
            abstract = id_to_abstact_map[doc_id]
            usefull = []
            for indexes in [evidence["sentences"] for evidence in evidence_sets]:
                usefull.extend(indexes)
            usefull_sentences = [abstract[index] for index in usefull]
            sentences.extend(usefull_sentences)

        for sentence in sentences:
            relevant_data.append((claim["claim"], sentence, 1.0))

    return relevant_data


def create_scifact_not_relevant(claim_path, corpus_path, k, embedding_model):
    not_relevant_data = []
    id_to_abstract_map = create_id_to_abstract_map(corpus_path)

    print("Embedding sentences to create fine-tune dataset")
    embedding_model = SentenceTransformer(embedding_model)
    embedded_id_to_abstract_map = dict()
    for doc_id, abstract in tqdm(id_to_abstract_map.items()):
        embedded_id_to_abstract_map[doc_id] = embedding_model.encode(abstract)

    for claim in tqdm(jsonlines.open(claim_path)):
        chosen_sentences = []

        negative_abstracts = [
            abstract
            for doc_id, abstract in id_to_abstract_map.items()
            if doc_id not in claim["evidence"]
        ]

        # make not_relevant datapoint from random abstract not used for evidence
        negative_sentences = []
        for abstract in sample(negative_abstracts, k):
            chosen_sentences = sample(abstract, 1)
            negative_sentences.extend(chosen_sentences)
        for sentence in negative_sentences:
            not_relevant_data.append((claim["claim"], sentence, 0.0))
            chosen_sentences.append(sentence)

        # make not_relevant for the abstrac with gold rationales
        evidence_obj = claim["evidence"]
        if evidence_obj:
            for doc_id, evidence in evidence_obj.items():
                abstract = id_to_abstract_map[doc_id]
                not_allowed = []
                for obj in evidence:
                    not_allowed.extend(obj["sentences"])
                not_allowed = set(not_allowed)
                allowed = [
                    abstract[index]
                    for index in set(range(0, len(abstract))) - not_allowed
                ]
                chosen = sample(allowed, min(2, len(allowed)))
                for sentence in chosen:
                    not_relevant_data.append((claim["claim"], sentence, 0.0))
                    chosen_sentences.append(sentence)
        # else use the abstract found in the cited_doc_ids value
        else:
            cited_doc_ids = claim["cited_doc_ids"]
            for abstract in [id_to_abstract_map[str(ident)] for ident in cited_doc_ids]:
                chosen = sample(abstract, min(2, len(abstract)))
                for sentence in chosen:
                    not_relevant_data.append((claim["claim"], sentence, 0.0))
                    chosen_sentences.append(sentence)

        # make not_relevant datapoint from the top 3 closest sentences to the claim (based on cosine similarity)
        top_3_closest_sentences_and_scores = []
        claim_embedding = embedding_model.encode(claim["claim"])
        for abstract, embedded_abstract in zip(id_to_abstract_map.values(), embedded_id_to_abstract_map.values()):
            for i, sentence_embedding in enumerate(embedded_abstract):
                cosine_similarity_score = get_cosine_similarities(claim_embedding, sentence_embedding)
                update_top_3_closest_sentences(cosine_similarity_score, abstract[i], top_3_closest_sentences_and_scores)

        for sentence, _ in top_3_closest_sentences_and_scores:
            if sentence not in chosen_sentences:
                not_relevant_data.append((claim["claim"], sentence, 0.0))

    return not_relevant_data


def update_top_3_closest_sentences(cosine_similarity_score, sentence, top_3_closest_sentences_and_scores):
    if len(top_3_closest_sentences_and_scores) < 3:
        top_3_closest_sentences_and_scores.append((sentence, cosine_similarity_score))

    scores = [score for _, score in top_3_closest_sentences_and_scores]
    min_score = min(scores)
    min_index = scores.index(min_score)

    if min_score < cosine_similarity_score:
        top_3_closest_sentences_and_scores[min_index] = (sentence, cosine_similarity_score)


def get_cosine_similarities(claim_embedding, sentence_embedding):
    return 1 - spatial.distance.cosine(claim_embedding, sentence_embedding)


def create_fever(claim_path):
    fever_data = []

    for claim in tqdm(jsonlines.open(claim_path)):
        # create a not relevant datapoint if not enough info
        if claim["label"] == "NOT ENOUGH INFO":
            allowed = [x for x in claim["sentences"] if x != ""]
            negative_evidence = sample(allowed, min(len(allowed), 2))
            for sentence in negative_evidence:
                fever_data.append((claim["claim"], sentence, 0.0))
            continue

        # clean sentences by removing the part with the references.
        cleaned_sentences = list(
            map(lambda a: a.split(" . ")[0] + ".", claim["sentences"])
        )
        # set label to 1 if the claim is supported by abstract, else 0
        # else create a relevant and not relevant datapoints from the sentence used to label
        for evidence in claim["evidence_sets"]:
            all_evidence = set()
            for index in evidence:
                all_evidence = all_evidence.union(set(evidence))

            # write to the relevant dataset
            usefull = [cleaned_sentences[index] for index in evidence]
            for sentence in usefull:
                fever_data.append((claim["claim"], sentence, 1.0))

            # write to the not_relevant dataset
            not_relevant_sents = set(range(len(cleaned_sentences))) - set(all_evidence)
            allowed = [
                cleaned_sentences[index]
                for index in not_relevant_sents
                if cleaned_sentences[index] != ""
            ]
            chosen = sample(allowed, min(2, len(allowed)))
            for sentence in chosen:
                fever_data.append((claim["claim"], sentence, 0.0))

    return fever_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a claims")
    parser.add_argument(
        "pretrained_model",
        metavar="path",
        type=str,
        help="the name of (or path to, if local) the pretrained sbert model",
    )
    parser.add_argument(
        "corpus",
        metavar="path",
        type=str,
        help="path to corpus.jsonl",
    )
    parser.add_argument(
        "train",
        metavar="path",
        type=str,
        help="path to train.jsonl",
    )
    parser.add_argument(
        "validation",
        metavar="path",
        type=str,
        help="path to validation.jsonl",
    )
    parser.add_argument(
        "fever_claims",
        metavar="path",
        type=str,
        help="path to fever_train.jsonl",
    )
    parser.add_argument(
        "output",
        metavar="path",
        type=str,
        help="path to the output path",
    )

    args = parser.parse_args()

    if os.path.exists(args.output):
        print("ERROR output folder exisits")
        exit()
    
    os.mkdir(args.output)

    create_training_data(args.corpus, args.train, args.fever_claims, args.pretrained_model, args.output)
    create_evaluator_data(args.corpus, args.validation, args.pretrained_model, args.output)
