import jsonlines
import numpy as np
from tqdm import tqdm
import models.sentence_selection.model as sentence_selection_module
import models.stance_prediction.model as stance_prediction_module
import time
import tensorflow as tf


def sentence_selection(claim, model, sentence_embeddings, corp_id, threshold):
    """Returns a dict that maps abstract ids to relevant sentences ids in that abstract
    i.e. {abstract_42: [{id: sent_3, embedding: <embedding>}, {id: sent_7, embedding: <embedding>}], abstract_127: [...]}
    We have at most 9 sentences per abstract
    """

    claim = tf.ones((sentence_embeddings.shape[0], 1)) * claim["claim"]
    claim_sent_embedding = tf.concat([claim, sentence_embeddings], 1)

    predicted = model(claim_sent_embedding)
    res_mask = tf.squeeze(tf.math.greater(predicted, tf.constant(threshold)))
    res = tf.where(res_mask)    

    sentence_scores_dict = dict()
    for pred_id in res:
        pred_id_val = pred_id[0]
        abstract_id, sentence_id = corp_id[pred_id_val]
        sentence_list = sentence_scores_dict.get(abstract_id, [])
        score = predicted[pred_id_val][0]
        sentence_list.append((score, sentence_id, pred_id_val))
        sentence_scores_dict[abstract_id] = sentence_list

    # Only use best 3 sentences
    relevant_sentences_dict = dict()
    for abstract_id, sentence_list in relevant_sentences_dict.items():
        sorted_by_score = sorted(sentence_list, key=lambda tup: tup[0], reverse=True)
        top_3_sentences_by_score = sorted_by_score[:3]
        correct_format = [{"id": sentence_id, "embedding": claim_sent_embedding[pred_id_val]} for _, sentence_id, pred_id_val in top_3_sentences_by_score]
        relevant_sentences_dict[abstract_id] = correct_format

    return relevant_sentences_dict


def same_prediction_as_avg(avg, pred, threshold):
    if avg < threshold and pred < threshold:
        return True
    elif avg >= threshold and pred >= threshold:
        return True
    return False


def stance_prediction(claim, evidence, model, output_writer=None):
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
        pred_sum = 0

        for sentence_dict in evidence[abstract]:
            embedding = tf.expand_dims(sentence_dict["embedding"], 0)
            pred = model(embedding)
            stance_predictions.append((sentence_dict["id"], pred))
            pred_sum += pred

        avg = pred_sum / len(stance_predictions)

        if output_writer is not None:
            output_writer.write("AVG predicted label value: {}".format(avg))

        threshold = tf.constant(0.5)
        rationale_sentences = [
            sent_id
            for sent_id, pred in stance_predictions
            if same_prediction_as_avg(avg, pred, threshold)
        ]
        label = "SUPPORT" if avg >= threshold else "CONTRADICT"
        resulting_evidence_dict[str(abstract)] = {
            "sentences": rationale_sentences,
            "label": label,
        }

    return {"id": claim_id, "evidence": resulting_evidence_dict}


def setup_sentence_embeddings(corpus_path):
    with jsonlines.open(corpus_path) as corpus_reader:
        corpus = np.array(list(corpus_reader.iter()))
    corp_id = []
    sentence_embeddings = []
    
    for line in corpus:
        for i in range(len(line['abstract'])):
            corp_id.append((line['doc_id'], i))
        sentence_embeddings.append(np.array(line['abstract']))

    sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
    return sentence_embeddings, corp_id


def run_pipeline(corpus_path, claims_path):
    threshold = 0.1
    abstract_retriever_model = sentence_selection_module.load()
    stance_prediction_model = stance_prediction_module.load() 
    sentence_embeddings, corp_id = setup_sentence_embeddings(corpus_path)
    with jsonlines.open("predictions.jsonl", "w") as output:
        with jsonlines.open(claims_path) as claims:
            for claim in tqdm(claims):
                relevant_sentences_dict = sentence_selection(claim, abstract_retriever_model, sentence_embeddings, corp_id, threshold)
                prediction = stance_prediction(claim, relevant_sentences_dict, stance_prediction_model) 
                output.write(prediction)


if __name__ == "__main__":
    corpus_path = "sbert-embedded-corpus.jsonl"
    claims_path = "sbert-embedded-dev-claims.jsonl"

    run_pipeline(corpus_path, claims_path)
