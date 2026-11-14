"""Exact external scoring with periods isolated by an ungraded separator."""
import numpy as np
from mtah.official import metrics


def evaluate_periods(truth, predictions):
    if set(truth) != set(predictions) or not truth:
        raise ValueError('PERIOD_MISMATCH')
    gt, pred = [], []
    for period in sorted(truth):
        a, b = np.asarray(truth[period]), np.asarray(predictions[period])
        if a.ndim != 1 or b.shape != a.shape:
            raise ValueError('FRAME_COUNT_MISMATCH')
        gt.extend([a, np.array(['X'])])
        pred.extend([b, np.array(['X'])])
    return metrics.evaluate(metrics.label_codes(np.concatenate(gt)),
                            metrics.label_codes(np.concatenate(pred)))
