from sentence_transformers import SentenceTransformer, util
import jsonlines
import time

class SpecterFilterModel():
    def __init__(self, corpus_path):
        #print("init")
        #t1 = time.time()
        self.model = SentenceTransformer('../../models/filter_corpus/specter_finetuned')
        self.corpus = list(jsonlines.open(corpus_path))
        mapped_corpus = [doc["title"] + " " + " ".join(doc['abstract']) for doc in self.corpus]
        #mapped_corpus = [sent for sent in doc['abstract'] for doc in self.corpus]
        self.corpus_embeddings = self.model.encode(mapped_corpus, convert_to_tensor=True)
        print(self.corpus_embeddings)
        #print("init done. Time:", time.time()-t1)

    def get_top_k_by_similarity(self, claim, corpus, k): 
        print("TODO")

    def get_top_k_by_similarity_with_ids(self, claim, corpus, corp_id, k, level):
        claim_embedding = self.model.encode(claim["claim"], convert_to_tensor=True)

        # v Bruger også cosine v
        search_hits = util.semantic_search(claim_embedding, self.corpus_embeddings, top_k = k)
        search_hits = search_hits[0]  #Get the hits for the first query (TODO: Man kan måske ordne alle claims samtidig, så)
        for hit in search_hits:
            print(hit)
        exit()
        
        doc_ids = []
        
        for idx, hit in enumerate(search_hits):
            if idx == k:
                continue

            doc = self.corpus[hit['corpus_id']]
            doc_ids.append(doc["doc_id"])
        
        return doc_ids
        

if __name__ == "__main__":
    SpecterFilterModel()