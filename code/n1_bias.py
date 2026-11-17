"""Fit a single N1 log-probability offset by minimizing duration MAE."""
from __future__ import annotations

import numpy as np

NON_N1 = np.asarray([0, 2, 3, 4], dtype=np.int8)


def probability_margin(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """N1 wins above log(max non-N1 p) - log(p_N1).

    At equality N1 wins over N2/N3/REM, but W wins over N1, matching
    numpy.argmax in the declared W,N1,N2,N3,REM class order.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 5 or not np.isfinite(p).all() or \
            np.any(p < 0) or np.any(p > 1) or np.any(p.sum(axis=1) <= 0):
        raise ValueError("Invalid five-class probabilities")
    other = NON_N1[np.argmax(p[:, NON_N1], axis=1)]
    max_other = p[np.arange(len(p)), other]
    with np.errstate(divide="ignore"):
        boundary = np.log(max_other) - np.log(p[:, 1])
    if np.isnan(boundary).any():
        raise ValueError("Undefined N1 margin")
    return boundary, other


def decode(probabilities: np.ndarray, bias: float) -> np.ndarray:
    """Apply the frozen log-probability offset to one night of predictions."""
    if not np.isfinite(bias):
        raise ValueError("Bias must be finite")
    boundary, other = probability_margin(probabilities)
    result = other.copy()
    n1 = (bias > boundary) | ((bias == boundary) & (other > 1))
    result[n1] = 1
    return result


def _minimize_duration_error(
    t: np.ndarray, exclusive: np.ndarray, night: np.ndarray,
    initial: np.ndarray, truth: np.ndarray,
) -> tuple[float, int, int, int]:
    """Sweep N1 switches and break MAE ties by distance to zero, then bias value."""
    n = len(truth)
    order = np.lexsort((exclusive, t))
    t, exclusive, night = t[order], exclusive[order], night[order]
    starts = np.flatnonzero(np.r_[True, (t[1:] != t[:-1]) |
                                 (exclusive[1:] != exclusive[:-1])])
    delta = np.empty(len(t), dtype=np.int8)
    for j in range(n):
        positions = np.flatnonzero(night == j)
        count_before = initial[j] + np.arange(len(positions))
        delta[positions] = np.where(count_before < truth[j], -1, 1)
    group_delta = np.add.reduceat(delta.astype(np.int64), starts)
    error_sum = np.r_[np.sum(np.abs(initial - truth)),
                      np.sum(np.abs(initial - truth)) + np.cumsum(group_delta)]
    group_t, group_exclusive = t[starts], exclusive[starts]
    # Each state spans the interval between consecutive switch groups.
    lower = np.r_[-np.inf, group_t]
    upper = np.r_[group_t, np.inf]
    lower_inclusive = np.r_[False, group_exclusive == 0]
    upper_inclusive = np.r_[group_exclusive == 1, False]
    lower_float = np.where(np.isfinite(lower) & ~lower_inclusive,
                           np.nextafter(lower, np.inf), lower)
    upper_float = np.where(np.isfinite(upper) & ~upper_inclusive,
                           np.nextafter(upper, -np.inf), upper)
    possible = lower_float <= upper_float
    representative = np.minimum(np.maximum(np.zeros_like(lower), lower_float), upper_float)
    possible &= np.isfinite(representative)
    best_error = np.min(error_sum[possible])
    optimal = np.flatnonzero(possible & (error_sum == best_error))
    preference = np.lexsort((representative[optimal], np.abs(representative[optimal])))
    winner = int(optimal[preference[0]])
    bias = float(representative[winner])
    return bias, int(best_error), int(possible.sum()), len(optimal)


def fit(calibration: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Fit the offset that minimizes N1-duration MAE across calibration nights."""
    if not calibration:
        raise ValueError("No calibration nights")
    boundaries, exclusives, night_ids = [], [], []
    initial, truth = [], []
    raw_error = 0
    epochs = 0
    for j, (p, reference) in enumerate(calibration):
        t, other = probability_margin(p)
        reference = np.asarray(reference)
        if reference.shape != (len(p),) or not np.isin(reference, np.arange(5)).all():
            raise ValueError("Invalid calibration reference")
        finite = np.isfinite(t)
        boundaries.append(t[finite])
        exclusives.append((other[finite] == 0).astype(np.int8))
        night_ids.append(np.full(np.sum(finite), j, dtype=np.int32))
        initial.append(int(np.sum(np.isneginf(t))))
        truth.append(int(np.sum(reference == 1)))
        raw_error += abs(int(np.sum(decode(p, 0) == 1)) - truth[-1])
        epochs += len(p)
    initial = np.asarray(initial, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    t = np.concatenate(boundaries)
    exclusive = np.concatenate(exclusives)
    night = np.concatenate(night_ids)
    n = len(calibration)
    if not len(t):
        return {"bias": 0.0, "calibration_mae_min": raw_error * .5 / n,
                "calibration_raw_mae_min": raw_error * .5 / n,
                "calibration_nights": n, "calibration_epochs": epochs,
                "prediction_states": 1, "optimal_states": 1}
    bias, best_error, states, optimal_states = _minimize_duration_error(
        t, exclusive, night, initial, truth,
    )
    decoded_error = sum(abs(int(np.sum(decode(p, bias) == 1)) -
                            int(np.sum(y == 1))) for p, y in calibration)
    if decoded_error != int(best_error):
        raise AssertionError(("Sweep/direct decode mismatch", bias, decoded_error, best_error))
    if decoded_error > raw_error:
        raise AssertionError("Optimized calibration error exceeds zero bias")
    return {"bias": bias, "calibration_mae_min": decoded_error * .5 / n,
            "calibration_raw_mae_min": raw_error * .5 / n,
            "calibration_nights": n, "calibration_epochs": epochs,
            "prediction_states": states, "optimal_states": optimal_states}
