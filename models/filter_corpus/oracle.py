class OracleFilterModel():
     def get_top_k_by_similarity_with_ids(self, claim, corpus, corp_id, k, level):
         return [int(abstract) for abstract in claim["evidence"].keys()]

