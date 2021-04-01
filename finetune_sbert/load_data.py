from tqdm import tqdm
import jsonlines
from random import sample

from sentence_transformers import InputExample
from sentence_transformers import evaluation


def load_training_data(scifact_claims_path, scifact_corpus_path, fever_claims_path):
    """ Both claim_path and corpus_path should be the text versions and not embedded versions """
    #return [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    #InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

    scifact_training_data = []

    scifact_relevant_data = create_scifact_relevant(scifact_claims_path, scifact_corpus_path)
    scifact_relevant_data_train_format = convert_to_train_format(scifact_relevant_data)
    scifact_training_data.extend(scifact_relevant_data_train_format)

    scifact_not_relevant_data = create_scifact_not_relevant(scifact_claims_path, scifact_corpus_path, 5)
    scifact_not_relevant_data_train_format = convert_to_train_format(scifact_not_relevant_data)
    scifact_training_data.extend(scifact_not_relevant_data_train_format)

    fever_data = create_fever(fever_claims_path)
    fever_data_train_format = convert_to_train_format(fever_data)

    return scifact_training_data, fever_data_train_format
    

def load_evaluator(validation_claims_path, corpus_path):
    #sentences1 = ['This list contains the first column', 'With your sentences', 'You want your model to evaluate on']
    #sentences2 = ['Sentences contains the other column', 'The evaluator matches sentences1[i] with sentences2[i]', 'Compute the cosine similarity and compares it to scores[i]']
    #scores = [0.3, 0.6, 0.2]

    claims = []
    sentences = []
    labels = []

    scifact_relevant_data = create_scifact_relevant(validation_claims_path, corpus_path)
    claims_relevant, sentences_relevant, labels_relevant = convert_to_evaluator_format(scifact_relevant_data)
    claims.extend(claims_relevant)
    sentences.extend(sentences_relevant)
    labels.extend(labels_relevant)

    scifact_not_relevant_data = create_scifact_not_relevant(validation_claims_path, corpus_path, 5)
    claims_not_relevant, sentences_not_relevant, labels_not_relevant = convert_to_evaluator_format(scifact_not_relevant_data)
    claims.extend(claims_not_relevant)
    sentences.extend(sentences_not_relevant)
    labels.extend(labels_not_relevant)

    return evaluation.EmbeddingSimilarityEvaluator(claims, sentences, labels)


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
            #relevant_data.append(InputExample(texts=[claim["claim"], sentence], label=1.0))
            relevant_data.append((claim["claim"], sentence, 1.0))

    return relevant_data


def create_scifact_not_relevant(claim_path, corpus_path, k):
    not_relevant_data = []

    id_to_abstact_map = create_id_to_abstract_map(corpus_path)

    for claim in tqdm(jsonlines.open(claim_path)):
        negative_abstracts = [
            abstract
            for doc_id, abstract in id_to_abstact_map.items()
            if doc_id not in claim["evidence"]
        ]

        # make not_relecant datapoint from random abstract not used for evidence
        negative_sentences = []
        for abstract in sample(negative_abstracts, k):
            chosen_sentences = sample(abstract, 1)
            negative_sentences.extend(chosen_sentences)
        for sentence in negative_sentences:
            not_relevant_data.append((claim["claim"], sentence, 0.0))
            #not_relevant_data.append(InputExample(texts=[claim["claim"], sentence], label=0.0))

        # make not_relevant for the abstrac with gold rationales
        evidence_obj = claim["evidence"]
        if evidence_obj:
            for doc_id, evidence in evidence_obj.items():
                abstract = id_to_abstact_map[doc_id]
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
                    #not_relevant_data.append(InputExample(texts=[claim["claim"], sentence], label=0.0))
        # else use the abstract found in the cited_doc_ids value
        else:
            cited_doc_ids = claim["cited_doc_ids"]
            for abstract in [id_to_abstact_map[str(ident)] for ident in cited_doc_ids]:
                chosen = sample(abstract, min(2, len(abstract)))
                for sentence in chosen:
                    not_relevant_data.append((claim["claim"], sentence, 0.0))
                    #not_relevant_data.append(InputExample(texts=[claim["claim"], sentence], label=0.0))

    return not_relevant_data


def create_fever(claim_path):
    fever_data = []

    for claim in tqdm(jsonlines.open(claim_path)):
        #claim_embedding = MODEL.encode(claim["claim"]).tolist()
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