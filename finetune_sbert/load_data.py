from tqdm import tqdm
import jsonlines
from random import sample

from sentence_transformers import InputExample
from sentence_transformers import evaluation


def load_training_data(claims_path, corpus_path):
    """ Both claim_path and corpus_path should be the text versions and not embedded versions """
    #return [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    #InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

    training_data = []

    # Create relevant pairs
    relevant_data = create_relevant(claims_path, corpus_path)

    # Create not relevant pairs
    not_relevant_data = create_not_relevant(claims_path, corpus_path, 5)

    training_data.extend(relevant_data)
    training_data.extend(not_relevant_data)
    return training_data
    

def create_id_to_abstract_map(corpus_path):
    abstract_id_to_abstract = dict()
    corpus = jsonlines.open(corpus_path)
    for data in corpus:
        abstract_id_to_abstract[str(data["doc_id"])] = data["abstract"]

    return abstract_id_to_abstract


def create_relevant(claims_path, corpus_path):
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
            relevant_data.append(InputExample(texts=[claim["claim"], sentence], label=1.0))

    return relevant_data


def create_not_relevant(claim_path, corpus_path, k):
    not_relevant_data = []

    id_to_abstact_map = create_id_to_abstract_map(corpus_path)

    for claim in tqdm(jsonlines.open(claim_path)):
        #claim_embedding = MODEL.encode(claim["claim"]).tolist()
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
            not_relevant_data.append(InputExample(texts=[claim["claim"], sentence], label=0.0))

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
                    not_relevant_data.append(InputExample(texts=[claim["claim"], sentence], label=0.0))
        # else use the abstract found in the cited_doc_ids value
        else:
            cited_doc_ids = claim["cited_doc_ids"]
            for abstract in [id_to_abstact_map[str(ident)] for ident in cited_doc_ids]:
                chosen = sample(abstract, min(2, len(abstract)))
                for sentence in chosen:
                    not_relevant_data.append(InputExample(texts=[claim["claim"], sentence], label=0.0))

    return not_relevant_data



def load_evaluator():
    sentences1 = ['This list contains the first column', 'With your sentences', 'You want your model to evaluate on']
    sentences2 = ['Sentences contains the other column', 'The evaluator matches sentences1[i] with sentences2[i]', 'Compute the cosine similarity and compares it to scores[i]']
    scores = [0.3, 0.6, 0.2]
    return evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
