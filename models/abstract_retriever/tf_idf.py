import jsonlines
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer



class TFIDFAbstractRetrieval:
    """ Uses TF-IDF and cosine similarity to extracttop k abstracts """

    def __init__(self, corpus_path, k=10, min_gram=1, max_gram=2):
        self.k = k
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(min_gram, max_gram)
        )
        self.corpus = list(jsonlines.open(corpus_path))
        self.doc_vectors = self.vectorizer.fit_transform(
            [doc["title"] + " ".join(doc["abstract"]) for doc in self.corpus]
        )
    
    def __call__(self, claim_object, abstracts):
        claim = claim_object["claim"]
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

        res = {int(doc_id): abstracts[doc_id] for doc_id in res}
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "corpus_path",
        metavar="path",
        type=str,
        help="the path to the sentence corpus",
    )
    parser.add_argument(
        "-st", "--sentence_threshold", type=float, default=0.5,
        help="the threshold for sentence selection"
    )

    args = parser.parse_args()

    t = TFIDFAbstractRetrieval(args.corpus_path, 10, 1, 2)
    print(t({"claim":"hello my friend"},{'1': ["nope"],'12489688': ["hello 1"], '13515165': ["hello 2"], '14682243': ["hello 3"], '198309074': ["hello 4"], '9988425': ["hello 5"], '9955779': ["hello 6"], '9956893': ["hello 7"], '9967265': ["hello 8"], '9973014': ["hello 9"], '9976969': ["hello 10"]}))

    