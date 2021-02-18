import argparse
import jsonlines
import os

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Create bar plots of evidence count")


def extract_number_of_rationale_sentences_in_evidence(evidence):
    res = []
    for _, rationales in evidence.items():
        for rationale in rationales:
            num = len(rationale["sentences"])
            res.append(num)
            # paper says rationales at most consists of 3 sentences.
            if num > 3:
                print(evidence)
    return res


# evidence_count_barchart creates a bar ploit of the number of evidences each claim
# have in a scifact data set
def evidence_count_barchart(dataset_path):
    counter = dict()
    with jsonlines.open(dataset_path) as data_set:
        for data_point in data_set:
            numbers = extract_number_of_rationale_sentences_in_evidence(
                data_point["evidence"]
            )
            for num in numbers:
                c = counter.get(num, 0) + 1
                counter[num] = c

    counter = list(counter.items())
    counter.sort(key=lambda a: a[0])
    unzipped = list(zip(*counter))

    num_of_evidences = list(map(str, unzipped[0]))
    count = list(unzipped[1])

    _, ax = plt.subplots(figsize=(8, 6))

    # Add title
    filename = os.path.basename(dataset_path)
    ax.set_title(
        "Number of rationales consisting of x sentences for file {}".format(filename)
    )
    # Add x, y gridlines
    ax.set_axisbelow(True)
    ax.grid(b=True, color="grey", linestyle="-", linewidth=0.5, alpha=0.2)
    # Create bar chart and set text describing the count
    ax.bar(num_of_evidences, count)
    padding = max(count) * 0.01
    for i in range(len(num_of_evidences)):
        plt.text(x=num_of_evidences[i], y=count[i] + padding, s=count[i], size=12)

    plt.show()


def main():
    parser.add_argument(
        "data_set_path", metavar="path", type=str, help="the path to the dataset"
    )
    args = parser.parse_args()
    evidence_count_barchart(args.data_set_path)


if __name__ == "__main__":
    main()
