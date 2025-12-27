from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int


def confusion_from_pairs(pairs: list[tuple[bool, bool]]) -> Confusion:
    """Compute confusion matrix from list of (predicted, actual) boolean pairs."""
    tp = fp = fn = tn = 0
    for predicted, actual in pairs:
        match (predicted, actual):
            case (True, True):
                tp += 1
            case (True, False):
                fp += 1
            case (False, True):
                fn += 1
            case (False, False):
                tn += 1
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


def _safe_div(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def precision_recall_f1(conf: Confusion) -> tuple[float, float, float]:
    """Compute precision, recall, and F1-score from a confusion matrix."""
    precision = _safe_div(conf.tp, conf.tp + conf.fp)
    recall = _safe_div(conf.tp, conf.tp + conf.fn)
    f1 = _safe_div(2 * conf.tp, 2 * conf.tp + conf.fp + conf.fn)  # equivalent to 2PR/(P+R)
    return precision, recall, f1
