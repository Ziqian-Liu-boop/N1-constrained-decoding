"""Limit Bias to W/N1/N2 transitions and use Full's nightly edit count."""
import numpy as np
from matched_bias import decode, fit


def fit_aligned(calibration_rows):
    """Fit the offset with up to 24 N1 additions and 24 removals per night."""
    return fit(calibration_rows, [24] * len(calibration_rows), restricted=True)


def apply_aligned(row, bias, full_labels):
    """Apply the fitted offset with Full's edit count as the candidate cap."""
    original = np.asarray(row["original"])
    full_labels = np.asarray(full_labels)
    if full_labels.shape != original.shape:
        raise ValueError("Full labels must match the original epoch sequence")
    limit = int(np.count_nonzero(full_labels != original))
    return decode(row, bias, limit, restricted=True)
