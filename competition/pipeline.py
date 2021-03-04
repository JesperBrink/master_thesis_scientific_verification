import jsonlines
import numpy as np
from tqdm import tqdm
from models.dense2LayerAbstractRetrieval.model import load
import time
import tensorflow as tf


def process_line(claim, line):
    abstract = np.array(line['abstract'])
    claim = np.ones((abstract.shape[0],1)) * np.array(claim)
    res = np.concatenate((claim, abstract), axis=1)
    return res


def find_relevant_sentences(claim, corpus, model):
    """ Returns a dict that maps abstract ids to relevant sentences ids in that abstract 
    i.e. {abstract_42: {sent_3, sent_7}, abstract_127: {sent_4}}
    """
    # TODO: potential speedup
    corp_id = []
    sentence_embeddings = []
    for line in corpus:
        for i in range(len(line['abstract'])):
            corp_id.append(line['doc_id'])
        sentence_embeddings.append(process_line(claim['claim'], line))
    sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
    # end todo
    
    predicted = model(sentence_embeddings)
    res_mask = tf.squeeze(tf.math.greater(predicted, tf.constant(0.99)))
    res = tf.where(res_mask)
    temp = [predicted[i[0]] for i in res]
    if len(temp):
        print(sum(temp) / len(temp))
    else:
        print("no corp found")
    gold = claim['evidence']
    found = [corp_id[i[0]] for i in res]
    print(claim['id'], len(found), [int(key) in found for key in gold.keys()])
    return None


def sentence_selection(claims_path, corpus_path):
    """ 
    input: all sentences in the corpus
    output: sentences that score above threshold (0.5) in being relevant
    """
    model = load()
    claim_id_to_rationale_dict = dict() # claim_id: {abstract_42: {sent_3, sent_7}, abstract_127: {sent_4}}
    corp = None
    with jsonlines.open(corpus_path) as corpus:
        corp = np.array(list(corpus.iter()))
    print("loaded model and started")

    with jsonlines.open(claims_path) as claims:
        for claim in tqdm(claims):
            relevant_sentences_dict = find_relevant_sentences(claim, corp, model)
            
            claim_id_to_rationale_dict[claim["id"]] = relevant_sentences_dict

    return claim_id_to_rationale_dict


def stance_prediction():
    """
    input: Claims + Rationales (from sentence selection)
    output: Whether abstracts/sentences support or refute claims
    """
    # gennemsnit eller alt ind på en gang?
    pass


def parse_to_output():
    # TODO: tjek hvordan vi skal skrive at vi ikke har et bud
    pass


if __name__ == '__main__':
    corpus_path = 'sbert-embedded-corpus.jsonl'
    claims_path= 'sbert-embedded-dev-claims.jsonl'
    
    claim_id_to_rationale_dict = sentence_selection(claims_path, corpus_path)
    # print(claim_id_to_rationale_dict)
    