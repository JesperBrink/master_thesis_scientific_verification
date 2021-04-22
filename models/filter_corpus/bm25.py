import jsonlines
from tqdm import tqdm
from pyserini.search import SimpleSearcher
import subprocess
import time
import os

class BM25FilterModel():
    def __init__(self, corpus_path, corpus_file_path, corpus_index_path, level, k1, b):
        self.map_corpus_to_pyserini_index(
            corpus_path, corpus_file_path, corpus_index_path, level)
        self.ssearcher = SimpleSearcher(corpus_index_path)
        self.ssearcher.set_bm25(k1, b)

    def get_top_k_by_similarity(self, claim, corpus, k):
        pass

    def get_top_k_by_similarity_with_ids(self, claim, corpus, corp_id, k, level):
        # doc_ids = set()

        # for abstract_id, sentence_number in corp_id:
        #     if level == "abstract":
        #         doc_ids.add(abstract_id)
        #     else:
        #         doc_ids.add((abstract_id, sentence_number))

        # scores = []

        # for doc in doc_ids:
        #     score = self.index_reader.compute_query_document_score(str(doc), claim["claim"])
        #     scores.append((score, doc))

        # sorted_scores = sorted(scores, key=lambda tup: tup[0], reverse=True)

        # return [doc[1] for doc in sorted_scores[:k]]
        bm25_hits = self.ssearcher.search(claim["claim"], k=k)
        return [int(hit.docid) for hit in bm25_hits]

    def map_corpus_to_pyserini_index(self, corpus_path, corpus_file_path, corpus_index_path, level):
        if os.path.exists(corpus_index_path):
            return

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
