from sentence_transformers import SentenceTransformer

MODEL = None

def s_bert_embed(doc, embedding_model):
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(embedding_model)
    new_abstract_list = MODEL.encode(doc["claim"])
    doc["claim"] = new_abstract_list.tolist()
    return doc

    

# TODO: tilføj stopwords
# allenai-specter, paraphrase-distilroberta-base-v1
# skriv noter til resultaterne, så vi kan smide det med i specialet