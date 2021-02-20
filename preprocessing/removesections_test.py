from utils.testutils import assertion
from removesections import remove_sections


def test_remove_sections():
    testcases = [
        {
            "name": "whitelisted section",
            "expected": {"abstract": ["RESULTS test sentence"], "structured": True},
            "input_claim": {"abstract": ["RESULTS test sentence"], "structured": True},
        },
        {
            "name": "non-whitelisted section",
            "expected": {"abstract": [""], "structured": True},
            "input_claim": {"abstract": ["BACKGROUND test sentence"], "structured": True},
        },
        {
            "name": "section name contains whitelisted section",
            "expected": {"abstract": [""], "structured": True},
            "input_claim": {"abstract": ["RESULTS AND CONCLUSIONS test sentence"], "structured": True},
        },
        {
            "name": "Unstructured abstracts",
            "expected": {"abstract": ["test sentence"], "structured": False},
            "input_claim": {"abstract": ["test sentence"], "structured": False},
        },
    ]
    for test in testcases:
        res = remove_sections(test["input_claim"])
        assertion(test["name"], test["expected"], res)


if __name__ == "__main__":
    test_remove_sections()
