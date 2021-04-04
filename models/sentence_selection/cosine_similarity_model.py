import tensorflow as tf

class CosineSimilaritySentenceSelector():
    def __call__(self, claim_sent_embeddings):
        claim_embedding = claim_sent_embeddings[0, :768]
        sentence_embeddings = claim_sent_embeddings[:,768:]
        
        return tf.reshape(get_cosine_similarities(claim_embedding, sentence_embeddings), (sentence_embeddings.shape[0], 1))

def get_cosine_similarities(claim_embedding, sentence_embeddings):
    norm_sentences = tf.math.l2_normalize(sentence_embeddings, 1)
    norm_claim = tf.math.l2_normalize(claim_embedding, 0)
    similarities = tf.linalg.matvec(norm_sentences, norm_claim)

    return similarities