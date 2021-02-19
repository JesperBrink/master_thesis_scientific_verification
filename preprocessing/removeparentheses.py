import re

def remove_parentheses(abstract, num_of_spaces):
    exp = r"[ ]\(([^\) ]*[ ]){" + str(num_of_spaces) + r",}[^\) ]*\)"
    res = []
    for sentence in abstract:
        res.append(re.sub(exp, "", sentence))
    return res
