import re


def remove_parentheses(doc, num_of_spaces):
    exp = r"[ ]\(([^\) ]*[ ]){" + str(num_of_spaces) + r",}[^\) ]*\)"
    for sentence_idx, sentence in enumerate(doc["abstract"]):
        doc["abstract"][sentence_idx] = re.sub(exp, "", sentence)
    return doc
