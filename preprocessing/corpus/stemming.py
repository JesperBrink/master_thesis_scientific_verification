from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

def stemming(doc):
    # NOTE: Also tokenizes in order to split words and punctuation
    stemmer = SnowballStemmer("english")
    for sentence_idx, sentence in enumerate(doc["abstract"]):
        doc["abstract"][sentence_idx] = " ".join(
                [stemmer.stem(w) for w in word_tokenize(sentence)]
            )

    return doc