from sentence_transformers import SentenceTransformer
import time

MODEL = None


def s_bert_embed(doc):
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer("stsb-distilbert-base")
    start = time.time()
    new_abstract_list = MODEL.encode(doc["abstract"])
    end = time.time()
    print(end - start)
    doc["abstract"] = new_abstract_list.tolist()
    return doc