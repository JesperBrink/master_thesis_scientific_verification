import argparse
import jsonlines
import tensorflow as tf
import numpy as np


class CosineSimilaritySentenceSelector:
    def __init__(self, corpus_embedding_path, claim_embedding_path, threshold=0.5, k=5):
        self.threshold = threshold
        self.k = k
        self.id_to_claim_embedding_map = self.create_id_to_claim_map(
            claim_embedding_path
        )
        self.sentence_embeddings, self.rationale_id_to_abstract_and_sentence_id_pair = self.setup_sentence_embeddings(corpus_embedding_path)

    def __call__(self, claim_object, abstracts):
        result = {}

        claim_id = claim_object["id"]
        claim_embedding = tf.constant(self.id_to_claim_embedding_map[claim_id])

        predicted = self.get_cosine_similarity(claim_embedding, self.sentence_embeddings)
        results_above_threshold_mask = tf.squeeze(tf.math.greater(predicted, tf.constant(self.threshold)))
        indices_for_above_threshold = tf.where(results_above_threshold_mask)        
        
        rationale_index_and_score_pairs = [(idx[0], predicted[idx[0]]) for idx in indices_for_above_threshold] 
        rationale_index_and_score_pairs_sorted_by_score = sorted(rationale_index_and_score_pairs, key=lambda tup: tup[1], reverse=True)

        selected_rationales = 0
        index = 0
        while selected_rationales < self.k and index < len(rationale_index_and_score_pairs_sorted_by_score):
            rationale_idx, score = rationale_index_and_score_pairs_sorted_by_score[index]
            abstract_id, sentence_id = self.rationale_id_to_abstract_and_sentence_id_pair[rationale_idx]
            abstract_rationales = result.setdefault(abstract_id, [])
            if len(abstract_rationales) < 3:
                abstract_rationales.append(sentence_id)
                result[abstract_id] = abstract_rationales
                selected_rationales += 1
            index += 1

        return result
    
    def get_cosine_similarity(self, claim_embedding, sentence_embeddings):
        norm_sentences = tf.math.l2_normalize(sentence_embeddings, 1)
        norm_claim = tf.math.l2_normalize(claim_embedding, 0)
        norm_sentences = tf.cast(norm_sentences, tf.float32)
        similarities = tf.linalg.matvec(norm_sentences, norm_claim)
        return tf.reshape(similarities, (sentence_embeddings.shape[0], 1))

    def create_id_to_claim_map(self, claim_path):
        claim_id_to_embeding = dict()
        claims = jsonlines.open(claim_path)
        for data in claims:
            claim_id_to_embeding[data["id"]] = data["claim"]

        return claim_id_to_embeding

    def setup_sentence_embeddings(self, corpus_path):
        with jsonlines.open(corpus_path) as corpus_reader:
            corpus = np.array(list(corpus_reader.iter()))
        rationale_id_to_abstract_and_sentence_id_pair = []
        sentence_embeddings = []
        
        for line in corpus:
            for i in range(len(line['abstract'])):
                rationale_id_to_abstract_and_sentence_id_pair.append((line['doc_id'], i))
            sentence_embeddings.append(np.array(line['abstract']))

        sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
        return sentence_embeddings, rationale_id_to_abstract_and_sentence_id_pair


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-cl", "--claim_embedding", type=str)
    parser.add_argument("-co", "--corpus_embedding", type=str)

    args = parser.parse_args()

    selector = CosineSimilaritySentenceSelector(
        args.corpus_embedding, args.claim_embedding, threshold=0.2
    )
    abstracts = {4983: [""]}
    print(selector({"id": 13, "claim": "gd is not"}, abstracts))


if __name__ == "__main__":
    main()
