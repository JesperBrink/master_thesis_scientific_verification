from sentence_transformers import SentenceTransformer


def s_bert_embed(doc):
    model = SentenceTransformer('stsb-distilbert-base')
    new_abstract_list = model.encode(doc['abstract'])
    doc['abstract'] = new_abstract_list
    return doc