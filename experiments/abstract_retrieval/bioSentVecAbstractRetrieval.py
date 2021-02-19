from sklearn.metrics.pairwise import cosine_similarity
import pickle


class BioSentVecAbstractRetrieval:
    """ Retrieves the k best  """

    def __init__(self, k, claims_embedding_pickle, corpus_embedding_pickle):
        self.k = k
        with open(corpus_embedding_pickle, "rb") as f:
            self.corpus_embeddings = pickle.load(f)
        with open(claims_embedding_pickle, "rb") as f:
            self.claim_embeddings = pickle.load(f)

    def retrieve(self, claim_id, _):
        abstract_and_similarity_pairs = []
        claim_embedding = self.claim_embeddings[claim_id]

        for abstract_id, abstract_embedding in self.corpus_embeddings.items():
            claim_similarity = cosine_similarity(claim_embedding, abstract_embedding)
            abstract_and_similarity_pairs.append((abstract_id, claim_similarity))

        abstract_and_similarity_pairs.sort(key=lambda tup: tup[1], reverse=True)
        abstracts_sorted_by_similarity, _ = zip(*abstract_and_similarity_pairs)

        return abstracts_sorted_by_similarity[: self.k]
