"""Auxiliary functions."""


def iou_metric(ground_truth: list, predictions: list) -> float:
    """
    Вычисляет метрику Intersection Over Union = |A \cap B| / |A \cup B|.

    Args:
        ground_truth (list): истинные значения
        predictions (list): предсказания

    Returns:
        float: iou метрика
    """
    if len(ground_truth) == 0:
        return 1 / (1 + len(predictions))
    elif len(predictions) == 0:
        return 0

    iou = len(set.intersection(set(ground_truth), set(predictions)))
    iou = iou / (len(set(ground_truth).union(set(predictions))))
    return iou
