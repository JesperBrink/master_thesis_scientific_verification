def _safe_divide(numerator, denominator):
    return 0 if denominator == 0 else numerator / denominator


def compute_precision(true_positives, false_positives):
    return _safe_divide(true_positives, true_positives + false_positives)


def compute_recall(true_positives, false_negatives):
    return _safe_divide(true_positives, true_positives + false_negatives)


def compute_f1(precision, recall):
    return 2 * _safe_divide(precision * recall, precision + recall)
