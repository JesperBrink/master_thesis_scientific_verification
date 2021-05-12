from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import jsonlines
import time

class SpecterFilterModel():
    def __init__(self, corpus_path):
        self.model = AutoModel.from_pretrained('allenai/specter')
        #self.model2 = SentenceTransformer('allenai/specter')
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.corpus = list(jsonlines.open(corpus_path))
        mapped_corpus = [doc["title"] + self.tokenizer.sep_token + " ".join(doc['abstract']) for doc in self.corpus]
        #self.corpus_embeddings2 = self.model2.encode(mapped_corpus, convert_to_tensor=True)
        self.inputs = self.tokenizer(mapped_corpus, padding=True, truncation=True, return_tensors="pt", max_length=512)
        self.results = self.model(**self.inputs)
        self.corpus_embeddings = self.results.last_hidden_state[:, 0, :]

    def get_top_k_by_similarity(self, claim, corpus, k): 
        print("TODO")

    def get_top_k_by_similarity_with_ids(self, claim, corpus, corp_id, k, level):
        #claim_embedding2 = self.model2.encode(claim["claim"], convert_to_tensor=True)
        inp = self.tokenizer(claim["claim"], padding=True, truncation=True, return_tensors="pt", max_length=512)
        res = self.model(**inp)
        claim_embedding = res.last_hidden_state[:, 0, :]

        # v Bruger også cosine v
        search_hits = util.semantic_search(claim_embedding, self.corpus_embeddings, top_k = k)
        search_hits = search_hits[0]  #Get the hits for the first query (TODO: Man kan måske ordne alle claims samtidig, så)
        #search_hits2 = util.semantic_search(claim_embedding2, self.corpus_embeddings2, top_k = k)
        #search_hits2 = search_hits2[0]  #Get the hits for the first query (TODO: Man kan måske ordne alle claims samtidig, så)
        # print(search_hits2)
        # for hit1, hit2 in zip(search_hits, search_hits2):
        #     print("torch",hit1, self.corpus[hit1['corpus_id']]['doc_id'])
        #     print("sbert",hit2, self.corpus[hit2['corpus_id']]['doc_id'])
        #     print("")
        # exit()
        doc_ids = []
        
        for idx, hit in enumerate(search_hits):
            if idx == k:
                continue

            doc = self.corpus[hit['corpus_id']]
            doc_ids.append(doc["doc_id"])
        
        return doc_ids