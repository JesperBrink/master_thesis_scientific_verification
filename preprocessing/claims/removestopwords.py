import nltk
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))


def remove_stopwords(doc):
    doc["claim"] = " ".join(
        [w for w in doc["claim"].split() if w.lower() not in STOPWORDS]
    )
    return doc
