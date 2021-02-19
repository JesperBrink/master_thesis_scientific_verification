from utils.testutils import assertion
from removeparentheses import remove_parentheses


def test_remove_parentheses():
    testcases = [
        {
            "name": "no parentheses no removal",
            "expected": ["hello test 1 2 3"],
            "input_claim": ["hello test 1 2 3"],
            "input_num": 2,
        },
        {
            "name": "one parentheses no removal",
            "expected": ["hello (test) 1 2 3"],
            "input_claim": ["hello (test) 1 2 3"],
            "input_num": 1,
        },
        {
            "name": "one parentheses one removal",
            "expected": ["hello 2 3"],
            "input_claim": ["hello (test 1) 2 3"],
            "input_num": 1,
        },
        {
            "name": "real life test",
            "expected": [
                "In the posterior limb of the internal capsule, the mean apparent diffusion coefficients at both times were similar."
            ],
            "input_claim": [
                "In the posterior limb of the internal capsule, the mean apparent diffusion coefficients at both times were similar (1.2 versus 1.1 microm2/ms)."
            ],
            "input_num": 2,
        },
    ]
    for test in testcases:
        res = remove_parentheses(test["input_claim"], test["input_num"])
        assertion(test["name"], test["expected"], res)


if __name__ == "__main__":
    test_remove_parentheses()
