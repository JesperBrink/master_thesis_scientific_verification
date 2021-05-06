import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_results(result_folder, thresholds, values_of_k):
    threshold_to_score_list_dict = dict()

    for threshold in thresholds:
        threshold_results = []
        for k in values_of_k:
            res_file = "k-{}/{}.json".format(k, threshold)
            path = os.path.join(result_folder, res_file)
            with open(path) as json_file:
                data = json.load(json_file)
                threshold_results.append(float(data["sentence_selection_f1"]) * 100)
        threshold_to_score_list_dict[threshold] = threshold_results

    return threshold_to_score_list_dict


def convert_threshold_to_label(threshold):
    return "Threshold: 0.{}".format(threshold[1])


def make_plot(values_of_k, threshold_to_score_list_dict, title):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for threshold, f1_scores in threshold_to_score_list_dict.items():
        ax.plot(values_of_k, f1_scores, label=convert_threshold_to_label(threshold))

    plt.xlabel("k", size=15)
    plt.ylabel("F1", size=15)

    plt.title(title, size=15)
    plt.legend()

    plt.show()


def main(result_folder, title):
    thresholds = ["05", "06", "07", "08"]
    values_of_k = [3, 4, 5, 6, 7, 8, 9, 10]
    threshold_to_score_list_dict = load_results(result_folder, thresholds, values_of_k)
    make_plot(values_of_k, threshold_to_score_list_dict, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create plots of F1 score for different values of k for the overall-top-k idea"
    )
    parser.add_argument(
        "result_folder", metavar="path", type=str, help="the path to the results"
    )
    parser.add_argument("title", type=str, help="the title of the plot")
    args = parser.parse_args()

    main(args.result_folder, args.title)
