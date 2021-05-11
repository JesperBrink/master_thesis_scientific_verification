import jsonlines
from tqdm import tqdm
from pyserini.search import SimpleSearcher
from pyserini.dsearch import SimpleDenseSearcher
from pyserini.dsearch import TCTColBERTQueryEncoder
from pyserini.hsearch import HybridSearcher
import subprocess
import time
import os
import faiss
import numpy as np
from transformers import BertModel, BertTokenizer
import json

class HybridFilterModel():
    def __init__(self, corpus_path, corpus_file_path, corpus_index_path, level):
        self.faiss_index = "faiss_index"
        self.map_corpus_to_pyserini_index(
            corpus_path, corpus_file_path, corpus_index_path, level)
        self.map_corpus_to_faiss_index(self.faiss_index, corpus_file_path) # TODO: Run again with title
        self.ssearcher = SimpleSearcher(corpus_index_path)
        self.encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco") # TODO: Find better model? Hvis man bare kan give selve embeddings til QueryEncoder, er det bedre, hvis vi træner vores egen
        self.dsearcher = SimpleDenseSearcher(self.faiss_index, self.encoder)
        self.hsearcher = HybridSearcher(self.dsearcher, self.ssearcher)

    def get_top_k_by_similarity(self, claim, corpus, k):
        pass

    def get_top_k_by_similarity_with_ids(self, claim, corpus, corp_id, k, level):
        # TODO: Just use hybrid, the rest is for testing
        hits1 = self.hsearcher.search(claim["claim"], k=k) # hybrid
        hits2 = self.ssearcher.search(claim["claim"], k=k) # bm25
        hits3 = self.dsearcher.search(claim["claim"], k=k) # dense. Hvad er det for en measure, de bruger??? Den giver sådan 50 i score
        return [int(hit.docid) for hit in hits2]


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

    def map_corpus_to_faiss_index(self, faiss_index, corpus_file_path):
        # create FAISS index
        if os.path.exists(faiss_index):
            return

        encoder = "bert-base-uncased" # TODO: DistilBert?
        dimension = 768
        corpus = corpus_file_path
        device = "cpu"

        tokenizer = BertTokenizer.from_pretrained(encoder)
        model = BertModel.from_pretrained(encoder)
        model.to(device)

        index = faiss.IndexFlatIP(dimension)

        if not os.path.exists(faiss_index):
            os.mkdir(faiss_index)

        with open(os.path.join(faiss_index, 'docid'), 'w') as id_file:
            for file in sorted(os.listdir(corpus)):
                file = os.path.join(corpus, file)
                #if file.endswith('json'):
                print(f'Encoding {file}')
                with open(file, 'r') as corpus:
                    for idx, line in enumerate(tqdm(corpus.readlines())):
                        info = json.loads(line)
                        docid = info['id']
                        text = info['contents']
                        id_file.write(f'{docid}\n')
                        embedding = self.encode_passage(text, tokenizer, model, device)
                        index.add(np.array([embedding]))
        faiss.write_index(index, os.path.join(faiss_index, 'index'))


    def encode_passage(self, text, tokenizer, model, device='cuda:0'):
        max_length = 154  # hardcode for now
        inputs = tokenizer(
            '[CLS] [D] ' + text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        inputs.to(device)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.detach().cpu().numpy()
        return np.mean(embeddings[:, 4:, :], axis=-2).flatten()
