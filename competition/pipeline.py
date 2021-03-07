import jsonlines
import numpy as np
from tqdm import tqdm
import models.sentence_selection.model as sentence_selection_module
import models.stance_prediction.model as stance_prediction_module
import time
import tensorflow as tf


def sentence_selection(claim, model, sentence_embeddings, corp_id):
    """ Returns a dict that maps abstract ids to relevant sentences ids in that abstract 
    i.e. {abstract_42: [sent_3, sent_7], abstract_127: [sent_4]}
    We have at most 9 sentences per abstract
    """
    claim = np.ones((sentence_embeddings.shape[0],1)) * np.array(claim["claim"])
    claim_sent_embedding = np.concatenate((claim, sentence_embeddings), axis=1)

    predicted = model(claim_sent_embedding)
    res_mask = tf.squeeze(tf.math.greater(predicted, tf.constant(0.95)))
    res = tf.where(res_mask)
 
    relevant_sentences_dict = dict()
    for pred_id in res:
        abstract_id, sentence_id = corp_id[pred_id[0]] 
        sentence_list = relevant_sentences_dict.get(abstract_id, [])
        if len(sentence_list) == 9:
            print("Sentence list is larger than 9. FIX THIS")
            continue
        sentence_list.append({"id": sentence_id, "embedding": claim_sent_embedding[pred_id].tolist()})
        relevant_sentences_dict[abstract_id] = sentence_list

    return relevant_sentences_dict


def same_prediction_as_avg(avg, pred):
    if avg < 0.5 and pred < 0.5:
        return True
    elif avg >= 0.5 and pred >= 0.5:
        return True
    return False


def stance_prediction(claim, evidence, model):
    """
    input: Claims + Rationales (from sentence selection)
    output: Whether abstracts/sentences support or refute claims
    """
    claim_id = claim["id"]
    
    if not evidence:
        return {"id": claim_id, "evidence": {}}
    
    resulting_evidence_dict = dict()
    for abstract in evidence.keys():
        stance_predictions = []
        embeddings = []
        pred_sum = 0
        
        for sentence_dict in evidence[abstract]:
            embedding = tf.expand_dims(tf.convert_to_tensor(sentence_dict["embedding"]), 0)
            embeddings.append(embedding)

        dataset = tf.data.Dataset.from_tensor_slices(embeddings)
        predictions = model.predict(dataset)

        for pred in predictions:
            stance_predictions.append((sentence_dict["id"], pred))
            pred_sum += pred

        avg = pred_sum / len(stance_predictions)
        rationale_sentences = [sent_id for sent_id, pred in stance_predictions if same_prediction_as_avg(avg, pred)]
        label = "SUPPORT" if avg >= 0.5 else "CONTRADICT"
        resulting_evidence_dict[str(abstract)] = {"sentences": rationale_sentences, "label": label}

    return {"id": claim_id, "evidence": resulting_evidence_dict}


def run_pipeline(corpus_path, claims_path):
    abstract_retriever_model = sentence_selection_module.load()
    stance_prediction_model = stance_prediction_module.load() 
    with jsonlines.open(corpus_path) as corpus_reader:
        corpus = np.array(list(corpus_reader.iter()))
    corp_id = []
    sentence_embeddings = []
    
    for line in corpus:
        for i in range(len(line['abstract'])):
            corp_id.append((line['doc_id'], i)) # (id_of_abstract, sentence_id)
        sentence_embeddings.append(np.array(line['abstract']))

    sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)

    with jsonlines.open("predictions.jsonl", "w") as output:
        with jsonlines.open(claims_path) as claims:
            for claim in tqdm(claims):
                relevant_sentences_dict = sentence_selection(claim, abstract_retriever_model, sentence_embeddings, corp_id)
                prediction = stance_prediction(claim, relevant_sentences_dict, stance_prediction_model) 
                output.write(prediction)
    
if __name__ == '__main__':
    corpus_path = 'sbert-embedded-corpus.jsonl'
    claims_path = 'sbert-embedded-dev-claims.jsonl'
    
    run_pipeline(corpus_path, claims_path)
