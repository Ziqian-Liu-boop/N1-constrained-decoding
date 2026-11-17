"""Candidate selection and decoding for edit-limited N1 bias."""

import numpy as np

from n1_bias import _minimize_duration_error


def events(row, cap=None, restricted=False):
    p = np.asarray(row['probabilities'], dtype=float)
    o = row['original']
    other_cols = np.array([0, 2] if restricted else [0, 2, 3, 4])
    other = other_cols[p[:, other_cols].argmax(axis=1)]
    with np.errstate(divide='ignore'):
        t = np.log(p[np.arange(len(p)), other]) - np.log(p[:, 1])
    add = np.flatnonzero((o != 1) & (np.isin(o, (0, 2)) if restricted else True) & np.isfinite(t))
    remove = np.flatnonzero((o == 1) & np.isfinite(t))
    add = add[np.argsort(t[add], kind='stable')]
    remove = remove[np.argsort(-t[remove], kind='stable')]
    if cap is not None:
        add = add[:int(cap)]
        remove = remove[:int(cap)]
    ids = np.r_[remove, add]
    return (ids, t[ids], other[ids], int(np.sum(o == 1)) - len(remove))


def decode(row, bias, cap=None, restricted=False):
    ids, t, other, _ = events(row, cap, restricted)
    out = row['original'].copy()
    wins = (bias > t) | (bias == t) & (other > 1)
    out[ids] = np.where(wins, 1, other)
    return out


def fit(rows, caps, restricted):
    margins = []
    wake_ties = []
    night_ids = []
    initial = []
    truth = []
    for j, (row, cap) in enumerate(zip(rows, caps)):
        _, t, other, start = events(row, cap, restricted)
        margins.append(t)
        wake_ties.append((other == 0).astype(np.int8))
        night_ids.append(np.full(len(t), j, dtype=int))
        initial.append(start)
        truth.append(np.sum(row['y_true'] == 1))
    initial = np.array(initial)
    truth = np.array(truth)
    t, exclusive, night = (np.concatenate(arrays) for arrays in (margins, wake_ties, night_ids))
    if not len(t):
        return dict(bias=0.0, calibration_mae_min=float(np.mean(np.abs(initial - truth)) * 0.5), states=1)
    bias, best, states, _ = _minimize_duration_error(t, exclusive, night, initial, truth)
    actual = sum(
        abs(np.sum(decode(row, bias, cap, restricted) == 1) - np.sum(row['y_true'] == 1))
        for row, cap in zip(rows, caps)
    )
    assert actual == best
    raw = sum((abs(np.sum(row['original'] == 1) - np.sum(row['y_true'] == 1)) for row in rows))
    assert best <= raw
    return dict(bias=bias, calibration_mae_min=float(best * 0.5 / len(rows)), states=states)


def native_cap(row, policy):
    if policy == 'raw':
        return 0
    if policy == 'consensus_half24':
        return 24
    if policy.startswith('threshold'):
        return 48
    assert policy == 'ratio_only'
    h = int(np.sum(row['original'] == 1))
    s = float(row['probabilities'][:, 1].sum())
    return max(0, round(h + 0.5 * abs(s - h)) - h) if h / max(s, 1e-07) < 0.55 else 0
