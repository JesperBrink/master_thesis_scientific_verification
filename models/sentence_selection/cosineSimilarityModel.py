import argparse
import jsonlines
import tensorflow as tf


class CosineSimilaritySentenceSelector:
    def __init__(self, corpus_embedding_path, claim_embedding_path, threshold=0.5):
        self.threshold = threshold
        self.doc_id_to_abst_embedding_map = self.create_id_to_abstract_map(
            corpus_embedding_path
        )
        self.id_to_claim_embedding_map = self.create_id_to_claim_map(
            claim_embedding_path
        )

    def __call__(self, claim_object, abstracts):
        result = {}

        claim_id = claim_object["id"]
        claim_embedding = tf.constant(self.id_to_claim_embedding_map[claim_id])
        for doc_id, _ in abstracts.items():

            sentence_embeddings = tf.constant(self.doc_id_to_abst_embedding_map[doc_id])
            norm_sentences = tf.math.l2_normalize(sentence_embeddings, 1)
            norm_claim = tf.math.l2_normalize(claim_embedding, 0)
            similarities = tf.linalg.matvec(norm_sentences, norm_claim)

            top_k, indices = tf.math.top_k(similarities, k=3)
            res = tf.reshape(
                tf.gather(indices, tf.where(top_k > self.threshold), axis=0), (-1)
            )
            rationales = res.numpy().tolist()
            if len(rationales) < 1:
                continue
            result[doc_id] = rationales

        return result

    def create_id_to_abstract_map(self, corpus_path):
        abstract_id_to_abstract = dict()
        corpus = jsonlines.open(corpus_path)
        for data in corpus:
            abstract_id_to_abstract[data["doc_id"]] = data["abstract"]

        return abstract_id_to_abstract

    def create_id_to_claim_map(self, claim_path):
        claim_id_to_embeding = dict()
        claims = jsonlines.open(claim_path)
        for data in claims:
            claim_id_to_embeding[data["id"]] = data["claim"]

        return claim_id_to_embeding


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
