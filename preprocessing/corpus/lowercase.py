def lowercase(doc):
    for sentence_idx, sentence in enumerate(doc["abstract"]):
        doc["abstract"][sentence_idx] = " ".join(
            [w.lower() for w in sentence.split()]
        )
    return doc