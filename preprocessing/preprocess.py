import jsonlines
from removeparentheses import remove_parentheses
from functools import reduce


def execute(inp, function):
    return function(inp)


def preprocess(data_set_path, out_put_path, *options):
    with jsonlines.open(data_set_path) as data_set:
        with jsonlines.open(out_put_path, "w") as output:
            for data_point in data_set:
                abstract = data_point["abstract"]
                preprocessed_abstract = reduce(execute, options, abstract)
                data_point["abstract"] = preprocessed_abstract
                output.write(data_point)


if __name__ == "__main__":
    preprocess(
        "/Users/muggel/python/master_thesis_scientific_verification/datasets/scifact/corpus.jsonl",
        "temp.jsonl",
        lambda a: remove_parentheses(a, 2),
    )
