import argparse
import jsonlines
import enum
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

class Preprocessor(enum.Enum):
    Parantheses = "paren"
    Section = "section"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create bar plots of evidence count")
    parser.add_argument(
        "corpus_path", metavar="path", type=str, help="the path to the corpus"
    )
    parser.add_argument(
        "output_path", metavar="path", type=str, help="the path to the preprocessed output"
    )
    parser.add_argument(
        "preprocessors", metavar="path", type=Preprocessor, help="the path to the corpus", nargs="+"
    )

    args = parser.parse_args()
    preprocessors = []
    for preprocessor in args.preprocessors:
        if preprocessor == Preprocessor.Parantheses:
            preprocessors.append(lambda d: remove_parentheses(d, 2))
        elif preprocessor == Preprocessor.Section:
            preprocessors.append(lambda d: remove_sections(d))


    preprocess(
        args.corpus_path,
        args.output_path,
        *preprocessors
    )
