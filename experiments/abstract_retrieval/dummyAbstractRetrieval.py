import jsonlines

def dummy_abstract_retrieval(claim, corpus_path):
    """ Simply picks the first 3 abstracs each time """
    corpus_file = jsonlines.open(corpus_path)

    retrieved_abstracts = []
    for i, data in enumerate(corpus_file):
        retrieved_abstracts.append(str(data["doc_id"]))
        if i > 1000:
            break

    return retrieved_abstracts 