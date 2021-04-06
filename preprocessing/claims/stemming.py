from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

def stemming(doc):
    # NOTE: Also tokenizes in order to split words and punctuation
    stemmer = SnowballStemmer("english")
    doc["claim"] = " ".join(
        [stemmer.stem(w) for w in word_tokenize(doc["claim"])]
    )

    return doc