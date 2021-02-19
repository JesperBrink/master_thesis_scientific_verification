import jsonlines
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFAbstractRetrieval:
    """ Uses TF-IDF and cosine similarity to extracttop k abstracts """

    def __init__(self, k, corpus_path, min_gram, max_gram):
        self.k = k
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(min_gram, max_gram)
        )
        self.corpus = list(jsonlines.open(corpus_path))
        self.doc_vectors = self.vectorizer.fit_transform(
            [doc["title"] + " ".join(doc["abstract"]) for doc in self.corpus]
        )

    def retrieve(self, _, claim):
        # create vector for claim
        claim_vector = self.vectorizer.transform([claim]).todense()
        # compute cosine similarity to alle abstracts
        doc_scores = np.asarray(self.doc_vectors @ claim_vector.T).squeeze()
        # get index of the best cosines
        doc_indices_rank = doc_scores.argsort()[::-1].tolist()
        # extract the best k
        res = map(
            str, [self.corpus[idx]["doc_id"] for idx in doc_indices_rank[: self.k]]
        )
        return list(res)
