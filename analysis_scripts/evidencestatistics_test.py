from evidencestatistics import extract_number_of_rationale_sentences_in_evidence


def assertion(name, expected, res):
    try:
        assert res == expected
        print("PASS - test: {}".format(name))
    except AssertionError:
        print(
            "FAIL - test: {}\n\texpected {}\n\tgot      {}".format(name, expected, res)
        )


def test_extract_number_of_rationale_sentences_in_evidence():
    testcases = [
        {
            "name": "simpleTest",
            "expected": [1],
            "input": {"13734012": [{"sentences": [4], "label": "SUPPORT"}]},
        },
        {
            "name": "multiple rationales and one with multiple sentences",
            "expected": [2, 1],
            "input": {
                "14717500": [
                    {"sentences": [2, 5], "label": "SUPPORT"},
                    {"sentences": [7], "label": "SUPPORT"},
                ]
            },
        },
    ]
    for test in testcases:
        res = extract_number_of_rationale_sentences_in_evidence(test["input"])
        assertion(test["name"], test["expected"], res)


if __name__ == "__main__":
    test_extract_number_of_rationale_sentences_in_evidence()
