import tensorflow as tf

class CosineSimilaritySentenceSelector():
    def __call__(self, claim_sent_embeddings):
        sentence_embeddings = claim_sent_embeddings[:,768:]
        claim_embedding = claim_sent_embeddings[0, :768]
        
        norm_sentences = tf.math.l2_normalize(sentence_embeddings, 1)
        norm_claim = tf.math.l2_normalize(claim_embedding, 0)
        similarities = tf.reshape(tf.linalg.matvec(norm_sentences, norm_claim), (sentence_embeddings.shape[0], 1))

        return similarities