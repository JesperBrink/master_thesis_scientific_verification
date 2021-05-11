import jsonlines
from tqdm import tqdm
from pyserini.search import SimpleSearcher
import subprocess
import time
import os
from models.filter_corpus.specter import SpecterFilterModel
from sentence_transformers import SentenceTransformer, util

class HybridV2FilterModel():
    def __init__(self, corpus_path, corpus_file_path, corpus_index_path, level):
        self.map_corpus_to_pyserini_index(
            corpus_path, corpus_file_path, corpus_index_path, level)
        self.ssearcher = SimpleSearcher(corpus_index_path)
        self.specter = SentenceTransformer('../../models/filter_corpus/specter_finetuned')
        self.corpus = list(jsonlines.open(corpus_path))
        mapped_corpus = [doc["title"] + " " + " ".join(doc['abstract']) for doc in self.corpus[:5]]
        self.corpus_embeddings = self.specter.encode(mapped_corpus, convert_to_tensor=True, show_progress_bar=True)

    def get_top_k_by_similarity(self, claim, corpus, k):
        pass

    def get_top_k_by_similarity_with_ids(self, claim, corpus, corp_id, k, level):
        claim_embedding = self.specter.encode(claim["claim"], convert_to_tensor=True)
        bm25_hits = self.ssearcher.search(claim["claim"], k=k) # bm25
        specter_hits = util.semantic_search(claim_embedding, self.corpus_embeddings, top_k = k)
        specter_hits = specter_hits[0] # TODO: Lookup docid ligesom i specter
        self.hybrid_search(bm25_hits, specter_hits, self.corpus)
        exit()
        #return [int(hit.docid) for hit in hits2]

    def map_corpus_to_pyserini_index(self, corpus_path, corpus_file_path, corpus_index_path, level):

        if not os.path.exists(corpus_file_path):
            os.makedirs(corpus_file_path)
        
        if not os.path.exists(corpus_index_path):
            os.makedirs(corpus_index_path)

        corpus_file = open("{}/corpus.jsonl".format(corpus_file_path), 'w')
        writer = jsonlines.Writer(corpus_file)

        with jsonlines.open(corpus_path) as corpus:
            for document in tqdm(corpus):
                if level == "abstract":
                    doc_id = document["doc_id"]
                    contents = document["title"] + " " + " ".join(document["abstract"])
                    writer.write({"id": str(doc_id), "contents": contents})
                else:
                    for i, sentence in enumerate(document["abstract"]):
                        doc_id = str((document["doc_id"], i))
                        writer.write({"id": doc_id, "contents": sentence})

        writer.close()
        corpus_file.close()

        subprocess.call("python3.8 -m pyserini.index \
            -collection JsonCollection \
            -generator DefaultLuceneDocumentGenerator \
            -threads 1 \
            -input {} \
            -index {} \
            -storePositions \
            -storeDocvectors \
            -storeRaw"
            .format(corpus_file_path, corpus_index_path), 
        shell=True)
    

    def hybrid_search(self, sparse_results, dense_results, corpus, alpha = 0.1):
        # dense_hits = {hit.docid: hit.score for hit in dense_results}
        # sparse_hits = {hit.docid: hit.score for hit in sparse_results}
        # hybrid_result = []
        # min_dense_score = min(dense_hits.values())
        # min_sparse_score = min(sparse_hits.values())
        # for doc in set(dense_hits.keys()) | set(sparse_hits.keys()):
        #     if doc not in dense_hits:
        #         score = alpha * sparse_hits[doc] + min_dense_score
        #     elif doc not in sparse_hits:
        #         score = alpha * min_sparse_score + dense_hits[doc]
        #     else:
        #         score = alpha * sparse_hits[doc] + dense_hits[doc]
        #     hybrid_result.append(DenseSearchResult(doc, score))
        # return sorted(hybrid_result, key=lambda x: x.score, reverse=True)
        
        sparse_hits = {int(hit.docid): hit.score for hit in sparse_results}
        dense_hits = {self.corpus[hit['corpus_id']]["doc_id"]: hit["score"] for hit in dense_results}
        print(sparse_hits)
        print(dense_hits)
        pass
