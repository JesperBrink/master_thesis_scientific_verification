from utils.testutils import assertion
from removestopwords import remove_stopwords


def test_remove_sections():
    testcases = [
        {
            "name": "whitelisted test removal of common stopwords",
            "input_claim": {
                "abstract": [
                    "is a sentence where words will be removed",
                    "Polyhedral temporal organism",
                ]
            },
            "expected": {
                "abstract": ["sentence words removed", "Polyhedral temporal organism"]
            },
        },
    ]
    for test in testcases:
        res = remove_stopwords(test["input_claim"])
        assertion(test["name"], test["expected"], res)


if __name__ == "__main__":
    test_remove_sections()
