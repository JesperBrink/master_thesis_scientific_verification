import argparse
import jsonlines
import enum
from tqdm import tqdm
from embed import s_bert_embed
from functools import reduce
from removestopwords import remove_stopwords
from lemmatization import lemmatization
from lowercase import lowercase
from stemming import stemming



def execute(inp, function):
    return function(inp)


def preprocess(data_set_path, out_put_path, *options):
    with jsonlines.open(data_set_path) as data_set:
        with jsonlines.open(out_put_path, "w") as output:
            for data_point in tqdm(data_set):
                preprocessed_doc = reduce(execute, options, data_point)
                output.write(preprocessed_doc)


class Preprocessor(enum.Enum):
    StopWords = "stopwords"
    Lemmatization = "lemma"
    Lowercase = "lowercase"
    Stemming = "stem"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a claims")
    parser.add_argument(
        "claims_path", metavar="path", type=str, help="the path to the claims"
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
        if preprocessor == Preprocessor.StopWords:
            preprocessors.append(remove_stopwords)
        elif preprocessor == Preprocessor.Lemmatization:
            preprocessors.append(lemmatization)
        elif preprocessor == Preprocessor.Lowercase:
            preprocessors.append(lowercase)
        elif preprocessor == Preprocessor.Stemming:
            preprocessors.append(stemming)

    if args.embed:
        preprocessors.append(lambda d: s_bert_embed(d, args.embed))

    preprocess(args.claims_path, args.output_path, *preprocessors)
