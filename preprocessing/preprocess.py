import jsonlines
from removeparentheses import remove_parentheses
from removesections import remove_sections
from functools import reduce


def execute(inp, function):
    return function(inp)


def preprocess(data_set_path, out_put_path, *options):
    with jsonlines.open(data_set_path) as data_set:
        with jsonlines.open(out_put_path, "w") as output:
            for data_point in data_set:
                preprocessed_doc = reduce(execute, options, data_point)
                output.write(preprocessed_doc)


if __name__ == "__main__":
    preprocess(
        "../datasets/scifact/corpus.jsonl",
        "temp.jsonl",
        lambda d: remove_parentheses(d, 2),
    )
