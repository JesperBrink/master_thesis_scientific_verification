import jsonlines


class DummyAbstractRetrieval:
    """ Simply picks the first 1000 abstracts each time """

    def __init__(self, corpus_path):
        corpus_file = jsonlines.open(corpus_path)
        retrieved_abstracts = []
        for i, data in enumerate(corpus_file):
            retrieved_abstracts.append(str(data["doc_id"]))
            if i > 1000:
                break
        self.result = retrieved_abstracts

    def retrieve(self, claim):
        return self.result
