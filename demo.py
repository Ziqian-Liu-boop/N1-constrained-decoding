"""Generate synthetic predictions and reference labels for the example."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import numpy as np

from n1_site_decoder import BASES


def write_night(folder: Path, subject: str, number: int, reference_additions: int, high_n1: bool) -> dict[str, str]:
    counts = (61, 6, 75, 12, 18) if high_n1 else (30, 6, 36, 12, 16)
    original = np.repeat(np.arange(5, dtype=np.int8), counts)
    candidates = np.flatnonzero(original == 2)[3:15]
    truth = original.copy()
    truth[candidates[:reference_additions]] = 1
    row = {"subject_id": subject}
    for model_index, base in enumerate(BASES):
        n1 = np.full(len(original), (0.04, 0.16)[model_index])
        winner = np.full(len(original), (0.84, 0.72)[model_index])
        n1[candidates] = np.linspace(0.179, 0.161, len(candidates)) - 0.001 * (number % 3)
        winner[candidates] = 0.46 + 0.01 * model_index
        if high_n1:
            eligible = np.isin(original, (0, 2))
            n1[eligible] = 0.45
            winner[eligible] = 0.46
        n1[original == 1] = 0.82
        winner[original == 1] = 0
        p = np.broadcast_to(((1 - n1 - winner) / 3)[:, None], (len(original), 5)).copy()
        p[:, 1] = n1
        non_n1 = np.flatnonzero(original != 1)
        p[non_n1, original[non_n1]] = winner[non_n1]
        p[original == 1] = (0.045, 0.82, 0.045, 0.045, 0.045)
        path = folder / base / f"{subject}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            path, epoch_indices=np.arange(len(original)), probabilities=p,
            original=original, y_true=truth,
        )
        row[base] = str(path)
    return row


def run_demo(
    evaluate: Callable[[Path, Path | None, Path, dict[str, int]], dict[str, Any]],
    settings: dict[str, int],
) -> dict[str, Any]:
    with TemporaryDirectory(prefix="n1-demo-") as temporary:
        directory = Path(temporary)
        calibration = [
            write_night(directory, f"cal_{i + 1:02}", i, 5 + i % 3, False) for i in range(20)
        ]
        test = [
            write_night(directory, f"test_{i + 1:02}", i, additions, i == 5)
            for i, additions in enumerate((5, 6, 8, 2, 4, 0))
        ]
        cal_file, test_file = directory / "calibration.json", directory / "test.json"
        cal_file.write_text(json.dumps(calibration), encoding="utf-8")
        test_file.write_text(json.dumps(test), encoding="utf-8")
        return evaluate(cal_file, None, test_file, settings)
