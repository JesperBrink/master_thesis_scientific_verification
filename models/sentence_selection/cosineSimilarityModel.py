import argparse
import jsonlines
import tensorflow as tf
import numpy as np
from sentence_transformers import CrossEncoder

class CosineSimilaritySentenceSelector:
    def __init__(self, corpus_embedding_path, claim_embedding_path, threshold=0.5, k=5, use_cross_encoder=False, corpus_path=None):
        self.threshold = threshold
        self.k = k
        self.id_to_claim_embedding_map = self.create_id_to_claim_map(
            claim_embedding_path
        )
        self.id_to_abstract_embedding_map = self.create_id_to_abstract_map(
            corpus_embedding_path
        )
        (
            self.sentence_embeddings,
            self.rationale_id_to_abstract_and_sentence_id_pair,
        ) = self.get_sentence_embeddings_for_all_abstracts(corpus_embedding_path)
        self.number_of_abstracts_in_corpus = self.get_number_of_abstracts_in_corpus(
            corpus_embedding_path
        )
        self.use_cross_encoder = use_cross_encoder
        if corpus_path:
            self.id_to_abstract_map = self.create_id_to_abstract_map(
                corpus_path
            )
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

    # Find de bedste ligesom før, men før vi vælger de bedste k, skal vi reranke dem med en cross-encoder
    # kan man bruge den SBERT vi har trænet som en BERT model --> der skal selvfølgelig trænes videre så der laves en specific version til det formål?

    def __call__(self, claim_object, retrieved_abstracts):
        result = {}

        claim_id = claim_object["id"]
        claim_embedding = tf.constant(self.id_to_claim_embedding_map[claim_id])

        if self.number_of_abstracts_in_corpus == len(retrieved_abstracts):
            sentence_embeddings = self.sentence_embeddings
            rationale_id_to_abstract_and_sentence_id_pair = self.rationale_id_to_abstract_and_sentence_id_pair
        else:
            sentence_embeddings, rationale_id_to_abstract_and_sentence_id_pair = self.get_sentence_embeddings_for_retreived_abstracts(retrieved_abstracts)

        predicted = self.get_cosine_similarity(claim_embedding, sentence_embeddings)
        results_above_threshold_mask = tf.squeeze(tf.math.greater(predicted, tf.constant(self.threshold)))
        indices_for_above_threshold = tf.where(results_above_threshold_mask)
        
        if self.use_cross_encoder:
            rationale_index_sorted_by_score = self.rerank_with_cross_encoder(
                claim_object,
                indices_for_above_threshold,
                rationale_id_to_abstract_and_sentence_id_pair
            )
        else:    
            rationale_index_sorted_by_score = self.sort_based_on_bi_encoder(
                indices_for_above_threshold,
                predicted
            )

        for rationale_idx in rationale_index_sorted_by_score[self.k]:
            abstract_id, sentence_id = rationale_id_to_abstract_and_sentence_id_pair[rationale_idx]
            abstract_rationales = result.setdefault(abstract_id, [])
            abstract_rationales.append(sentence_id)
            result[abstract_id] = abstract_rationales

        return result

    def rerank_with_cross_encoder(self, claim_obj, indices_for_above_threshold, rationale_id_to_abstract_and_sentence_id_pair):
        cross_encoder_input = []
        for rationale_idx in indices_for_above_threshold:
            abstract_id, sentence_id = rationale_id_to_abstract_and_sentence_id_pair[rationale_idx]
            sentence = self.id_to_abstract_map[abstract_id][sentence_id]
            cross_encoder_input.append((claim_obj["claim"], sentence))
        
        cross_scores = self.cross_encoder.predict(cross_encoder_input)

        rationale_index_and_score_pairs = [
            (rationale_idx, score) for rationale_idx, score in zip(indices_for_above_threshold, cross_scores)
        ]
        rationale_index_and_score_pairs_sorted_by_score = sorted(
            rationale_index_and_score_pairs, key=lambda tup: tup[1], reverse=True
        )
        return [idx for idx, _ in rationale_index_and_score_pairs_sorted_by_score]

    def sort_based_on_bi_encoder(self, indices_for_above_threshold, predicted):
        rationale_index_and_score_pairs = [
            (idx[0], predicted[idx[0]]) for idx in indices_for_above_threshold
        ]
        rationale_index_and_score_pairs_sorted_by_score = sorted(
            rationale_index_and_score_pairs, key=lambda tup: tup[1], reverse=True
        )
        return [idx for idx, _ in rationale_index_and_score_pairs_sorted_by_score]

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

    def create_id_to_abstract_map(self, corpus_path):
        abstract_id_to_abstract = dict()
        corpus = jsonlines.open(corpus_path)
        for data in corpus:
            abstract_id_to_abstract[data["doc_id"]] = data["abstract"]

        return abstract_id_to_abstract

    def get_sentence_embeddings_for_retreived_abstracts(self, retrieved_abstracts):
        rationale_id_to_abstract_and_sentence_id_pair = []
        sentence_embeddings = []

        for abstract_id in retrieved_abstracts.keys():
            abstract_embedding = self.id_to_abstract_embedding_map[abstract_id]
            for i in range(len(abstract_embedding)):
                rationale_id_to_abstract_and_sentence_id_pair.append((abstract_id, i))
            sentence_embeddings.append(np.array(abstract_embedding))

        sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
        return sentence_embeddings, rationale_id_to_abstract_and_sentence_id_pair

    def get_sentence_embeddings_for_all_abstracts(self, corpus_path):
        with jsonlines.open(corpus_path) as corpus_reader:
            corpus = np.array(list(corpus_reader.iter()))
        rationale_id_to_abstract_and_sentence_id_pair = []
        sentence_embeddings = []

        for line in corpus:
            for i in range(len(line["abstract"])):
                rationale_id_to_abstract_and_sentence_id_pair.append(
                    (line["doc_id"], i)
                )
            sentence_embeddings.append(np.array(line["abstract"]))

        sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
        return sentence_embeddings, rationale_id_to_abstract_and_sentence_id_pair

    def get_number_of_abstracts_in_corpus(self, corpus_path):
        with jsonlines.open(corpus_path) as corpus_reader:
            corpus = list(corpus_reader.iter())
            return len(corpus)


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
