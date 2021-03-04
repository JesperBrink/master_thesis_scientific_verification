import nltk
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))


def remove_stopwords(doc):
    for sentence_idx, sentence in enumerate(doc["abstract"]):
        doc["abstract"][sentence_idx] = " ".join(
            [w for w in sentence.split() if w.lower() not in STOPWORDS]
        )
    return doc
