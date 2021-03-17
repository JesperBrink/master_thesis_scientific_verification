from sentence_transformers import SentenceTransformer

MODEL = None

def s_bert_embed(doc):
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer("stsb-distilbert-base")
    new_abstract_list = model.encode(doc["claim"])
    doc["claim"] = new_abstract_list.tolist()
    return doc
