import jsonlines
from tqdm import tqdm
from pyserini.search import SimpleSearcher
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.search import querybuilder
from pyserini.pyclass import autoclass
import subprocess
import time
import os

class BM25FilterModel():
    def __init__(self, corpus_path, corpus_file_path, corpus_index_path, level, k1=0.9, b=0.4):
        self.map_corpus_to_pyserini_index(
            corpus_path, corpus_file_path, corpus_index_path, level)
        self.ssearcher = SimpleSearcher(corpus_index_path)
        self.ssearcher.set_bm25(k1, b)
        self.analyzer = Analyzer(get_lucene_analyzer())

    def get_top_k_by_similarity(self, claim, corpus, k):
        pass

    def get_top_k_by_similarity_with_ids(self, claim, corpus, corp_id, k, level):
        bm25_hits = self.ssearcher.search(claim["claim"], k=k)
        
        if level == "sentence":
            res = [(int(tup[0]), int(tup[1])) for hit in bm25_hits if (tup := hit.docid.split(","))]
        else:
            res = [int(hit.docid) for hit in bm25_hits]

        return res

    def map_corpus_to_pyserini_index(self, corpus_path, corpus_file_path, corpus_index_path, level):
        # if os.path.exists(corpus_index_path):
        #     return

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
                    #contents = " ".join(document["abstract"])
                    writer.write({"id": str(doc_id), "contents": contents})
                else:
                    for i, sentence in enumerate(document["abstract"]):
                        doc_id = str(document["doc_id"]) + "," + str(i)
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
