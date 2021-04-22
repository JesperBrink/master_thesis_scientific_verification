from pyserini.search import SimpleSearcher
from jnius import autoclass
import jsonlines
from tqdm import tqdm
import os


class BM25AbstractRetrieval:
    """ Uses Pyserini BM25 to extract top k abstracts """

    def __init__(self, corpus_path, k=36):
        self.k = k
        corpus_index_path = "bm25_abstract_index"  # Folder to write Pyserini index to
        self.map_corpus_to_pyserini_index(corpus_path, corpus_index_path)
        self.index_searcher = SimpleSearcher(corpus_index_path)
        self.index_searcher.set_bm25(
            1.0, 0.8
        )  # Default params - does this even need to be set?
        print("###  BM25 INIT DONE ###")

    def __call__(self, claim_object, abstracts):
        claim = claim_object["claim"]
        hits = self.index_searcher.search(claim, k=self.k)

        return {int(hit.docid): abstracts[int(hit.docid)] for hit in hits}

    def map_corpus_to_pyserini_index(self, corpus_path, corpus_index_path):
        mapped_corpus_path = "bm25_abstract_mapped_corpus"

        if not os.path.exists(corpus_index_path):
            os.makedirs(corpus_index_path)

        if not os.path.exists(mapped_corpus_path):
            os.makedirs(mapped_corpus_path)

        mapped_corpus_file = open("{}/corpus.jsonl".format(mapped_corpus_path), "w")
        writer = jsonlines.Writer(mapped_corpus_file)

        # Map SciFact corpus to the format Pyserini wants
        with jsonlines.open(corpus_path) as corpus:
            for doc in tqdm(corpus):
                doc_id = doc["doc_id"]
                contents = " ".join(doc["abstract"])
                writer.write({"id": str(doc_id), "contents": contents})

        writer.close()
        mapped_corpus_file.close()

        # Create Pyserini index (for the searcher) from mapped corpus
        args = [
            "-collection",
            "JsonCollection",
            "-generator",
            "DefaultLuceneDocumentGenerator",
            "-threads",
            "1",
            "-input",
            mapped_corpus_path,
            "-index",
            corpus_index_path,
            "-storePositions",
            "-storeDocvectors",
            "-storeRaw",
        ]
        JIndexCollection = autoclass("io.anserini.index.IndexCollection")
        JIndexCollection.main(args)
