import tensorflow as tf
from tqdm import tqdm
import time
from models.sentence_selection.cosine_similarity_model import get_cosine_similarities
from scipy.spatial import distance

class CosineSimilarityFilterModel():
    def get_top_k_by_similarity(self, claim, corpus_embeddings, k): 
        claim_embedding = tf.cast(claim["claim"], tf.float64)
        similarities = get_cosine_similarities(claim_embedding, corpus_embeddings)
        values, indices = tf.math.top_k(similarities, k, sorted=True)
        top_k_embeddings = tf.gather(corpus_embeddings, indices)
        
        return top_k_embeddings
    

    def get_top_k_by_similarity_with_ids(self, claim, corpus_embeddings, corp_id, k):
        claim_embedding = tf.cast(claim["claim"], tf.float64)
        similarities = get_cosine_similarities(claim_embedding, corpus_embeddings)
        values, indices = tf.math.top_k(similarities, k, sorted=True)
        top_k_sentences = tf.gather(corp_id, indices)

        return top_k_sentences
