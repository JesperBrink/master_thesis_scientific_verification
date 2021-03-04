from sentence_transformers import SentenceTransformer


def s_bert_embed(doc):
    model = SentenceTransformer("stsb-distilbert-base")
    new_abstract_list = model.encode(doc["claim"])
    doc['claim'] = new_abstract_list.tolist()
    return doc
