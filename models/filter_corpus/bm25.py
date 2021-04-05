import jsonlines
from rank_bm25 import BM25Okapi
import numpy as np

class BM25FilterModel():
    def __init__(self, corpus_path, level):
        self.corpus = []

        with jsonlines.open(corpus_path) as data:
            for document in data:
                if level == "abstract":
                    document["abstract"] = [" ".join(document["abstract"])]
                
                for sentence in document["abstract"]:
                    self.corpus.append(sentence.split(" "))
        self.bm25 = CustomBM25Okapi(self.corpus)

    def get_top_k_by_similarity(self, claim, corpus, k):
        pass

    def get_top_k_by_similarity_with_ids(self, claim, corpus, corp_id, k, level):
        tokenized_claim = claim["claim"].split(" ")
        top_k = self.bm25.get_top_n(tokenized_claim, self.corpus, corp_id, k, True)

        return top_k

class CustomBM25Okapi(BM25Okapi):
    def __init__(self, corpus, tokenizer=None):
        super().__init__(corpus, tokenizer)
    
    def get_top_n(self, query, documents, corp_id, n=5, get_ids=False):
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        documents = list(zip(corp_id, documents))
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        if get_ids:
            return [documents[i][0] for i in top_n]
        else:
            return [documents[i][1] for i in top_n]