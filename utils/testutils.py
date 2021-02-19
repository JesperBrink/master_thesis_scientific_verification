def assertion(name, expected, res):
    try:
        assert res == expected
        print("PASS - test: {}".format(name))
    except AssertionError:
        print(
            "FAIL - test: {}\n\texpected {}\n\tgot      {}".format(name, expected, res)
        )

