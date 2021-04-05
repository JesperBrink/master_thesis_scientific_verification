from nltk.stem import SnowballStemmer

def stemming(doc):
    stemmer = SnowballStemmer("english")
    for sentence_idx, sentence in enumerate(doc["abstract"]):
        doc["abstract"][sentence_idx] = " ".join(
                [stemmer.stem(w) for w in sentence.split()]
            )

    return doc