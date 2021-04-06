def lowercase(doc):
    doc["claim"] = " ".join(
        [w.lower() for w in doc["claim"].split()]
    )
    return doc