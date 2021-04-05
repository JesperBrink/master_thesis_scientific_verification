import argparse
import jsonlines
import enum
from tqdm import tqdm
from removeparentheses import remove_parentheses
from removesections import remove_sections
from removestopwords import remove_stopwords
from lemmatization import lemmatization
from lowercase import lowercase
from embed import s_bert_embed
from functools import reduce

# lowercasing (or is this handled during tokenization?)
# noise removal (e.g. unicode)?

def execute(inp, function):
    return function(inp)


def preprocess(data_set_path, out_put_path, *options):
    with jsonlines.open(data_set_path) as data_set:
        with jsonlines.open(out_put_path, "w") as output:
            for data_point in tqdm(data_set):
                preprocessed_doc = reduce(execute, options, data_point)
                output.write(preprocessed_doc)


class Preprocessor(enum.Enum):
    Parantheses = "paren"
    Section = "section"
    StopWords = "stopwords"
    Lemmatization = "lemma"
    Lowercase = "lowercase"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a corpus")
    parser.add_argument(
        "corpus_path", metavar="path", type=str, help="the path to the corpus"
    )
    parser.add_argument(
        "output_path",
        metavar="path",
        type=str,
        help="the path to the preprocessed output",
    )
    parser.add_argument(
        "preprocessors",
        metavar="preprocessors",
        type=Preprocessor,
        help="list of functions used for preprocessing",
        nargs="*",
    )
    parser.add_argument(
        "-e",
        "--embed",
        help="whether or not the preprocessing should end out in an embedding",
    )

    args = parser.parse_args()
    preprocessors = []
    for preprocessor in args.preprocessors:
        if preprocessor == Preprocessor.Parantheses:
            preprocessors.append(lambda d: remove_parentheses(d, 2))
        elif preprocessor == Preprocessor.Section:
            preprocessors.append(remove_sections)
        elif preprocessor == Preprocessor.StopWords:
            preprocessors.append(remove_stopwords)
        elif preprocessor == Preprocessor.Lemmatization:
            preprocessors.append(lemmatization)

    if args.embed:
        preprocessors.append(lambda d: s_bert_embed(d, args.embed))

    preprocess(args.corpus_path, args.output_path, *preprocessors)
