from scipy.spatial import distance
import tensorflow as tf
import time

class CosineSimilaritySentenceSelector():
    def __call__(self, claim_sent_embeddings):
        t1 = time.time()
        sentence_embeddings = claim_sent_embeddings[:,768:]
        claim_embedding = claim_sent_embeddings[0, :768]
        
        norm_sentences = tf.math.l2_normalize(sentence_embeddings, 1)
        norm_claim = tf.math.l2_normalize(claim_embedding, 0)
        similarities = tf.reshape(tf.linalg.matvec(norm_sentences, norm_claim), (sentence_embeddings.shape[0], 1)) # TODO: Map til tensor[[],[]]
        t2 = time.time()
        
        similarities_old = []
        #cosine_dist = tf.keras.losses.CosineSimilarity()
        # for claim_sent_embedding in claim_sent_embeddings:
        #     claim_embedding = claim_sent_embedding[:768]
        #     sentence_embedding = claim_sent_embedding[768:]
        #     claim_sent_similarity = -1 * cosine_dist(claim_embedding, sentence_embedding) # -1 because tf flips value as it is a loss, i.e. -1 is completely similar
        #     print(type(claim_sent_similarity))
        #     similarities.append([claim_sent_similarity])

        for claim_sent_embedding in claim_sent_embeddings:
            claim_embedding = claim_sent_embedding[:768]
            sentence_embedding = claim_sent_embedding[768:]
            claim_sent_similarity = 1-distance.cosine(claim_embedding, sentence_embedding)
            similarities_old.append([tf.constant(claim_sent_similarity, dtype=tf.float32)])
        t3 = time.time()
        print("new tf:", t2-t1)
        print("scipy:", t3-t2)

        return similarities

    # why do we do this? SBERT preprocess + FFNN sucks,  either we use SBERT as intended (preprocess + cosine), or we take a holistic approach and embed on runtime. This is the first of the two
    # The other ideas with SBERT + cosine and BM25 is to preprocess by only having most important abstracts or sentences, thus reducing the complexity for whatever model we use