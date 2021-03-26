from scipy.spatial import distance
import tensorflow as tf
import time

class CosineSimilaritySentenceSelector():
    def __call__(self, claim_sent_embeddings):
        similarities = []

        cosine_dist = tf.keras.losses.CosineSimilarity()
        #t1 = time.time()
        for claim_sent_embedding in claim_sent_embeddings:
            claim_embedding = claim_sent_embedding[:768]
            sentence_embedding = claim_sent_embedding[768:]
            claim_sent_similarity = cosine_dist(claim_embedding, sentence_embedding)
            similarities.append([claim_sent_similarity])
        #t2 = time.time()

        # for claim_sent_embedding in claim_sent_embeddings:
        #     claim_embedding = claim_sent_embedding[:768]
        #     sentence_embedding = claim_sent_embedding[768:]
        #     claim_sent_similarity = distance.cosine(claim_embedding, sentence_embedding)
        #     similarities.append([claim_sent_similarity])
        #t3 = time.time()
        #print(t2-t1)
        #print(t3-t2)

        return tf.convert_to_tensor(similarities)

    # why do we do this? SBERT preprocess + FFNN sucks,  either we use SBERT as intended (preprocess + cosine), or we take a holistic approach and embed on runtime. This is the first of the two
    # The other ideas with SBERT + cosine and BM25 is to preprocess by only having most important abstracts or sentences, thus reducing the complexity for whatever model we use