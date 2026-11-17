"""Evaluate N1 decoding on paired U-Sleep and DeepResNet predictions."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import aligned_bias
import n1_bias
from n1_site_decoder import BASES, Night, _read_manifest, apply_policy, select_with_expansion

METHODS = ("Original", "Full", "Unconstrained Bias", "Aligned Bias")
CONFIG = ROOT / "config" / "analysis.json"


def load_cohorts(
    initial_path: Path, expanded_path: Path | None, evaluation_path: Path,
) -> tuple[list[dict[str, Night]], list[dict[str, Night]], list[dict[str, Night]]]:
    initial, initial_subjects = _read_manifest(initial_path, with_reference=True)
    evaluation, evaluation_subjects = _read_manifest(evaluation_path, with_reference=False)
    if len(initial_subjects) != 20:
        raise ValueError("The initial calibration manifest must contain 20 subjects.")

    expanded, expanded_subjects = [], set()
    if expanded_path is not None:
        expanded, expanded_subjects = _read_manifest(expanded_path, with_reference=True)
        if not initial_subjects <= expanded_subjects or not 20 < len(expanded_subjects) <= 40:
            raise ValueError("Expansion must retain the initial subjects and contain 21–40 subjects.")
    initial_records = {pair[BASES[0]].record for pair in initial}
    expanded_records = {pair[BASES[0]].record for pair in expanded}
    evaluation_records = {pair[BASES[0]].record for pair in evaluation}
    if expanded and not initial_records <= expanded_records:
        raise ValueError("Expansion must retain every initial calibration night.")
    if (initial_subjects | expanded_subjects) & evaluation_subjects:
        raise ValueError("Calibration and evaluation subjects overlap.")
    if (initial_records | expanded_records) & evaluation_records:
        raise ValueError("Calibration and evaluation recordings overlap.")
    if not evaluation or len(evaluation_subjects) != len(evaluation):
        raise ValueError("The evaluation manifest must contain one night per subject.")
    return initial, expanded, evaluation


def as_row(night: Night) -> dict[str, np.ndarray]:
    return {"probabilities": night.probabilities, "original": night.original}


def fit_biases(calibration: list[dict[str, Night]]) -> dict[str, dict[str, float]]:
    fits = {}
    for base in BASES:
        nights = [pair[base] for pair in calibration]
        rows = [{**as_row(night), "y_true": night.reference} for night in nights]
        fits[base] = {
            "Unconstrained Bias": n1_bias.fit([(r["probabilities"], r["y_true"]) for r in rows])["bias"],
            "Aligned Bias": aligned_bias.fit_aligned(rows)["bias"],
        }
    return fits


def score_night(labels: np.ndarray, truth: np.ndarray, original: np.ndarray) -> dict[str, Any]:
    n1_signed_error = np.count_nonzero(labels == 1) - np.count_nonzero(truth == 1)
    confusion = np.bincount(5 * truth.astype(int) + labels.astype(int), minlength=25).reshape(5, 5)
    return {
        "signed_error_min": 0.5 * n1_signed_error,
        "edits": int(np.count_nonzero(labels != original)),
        "confusion": confusion,
    }


def summarize(rows: list[dict[str, Any]], original_error: np.ndarray) -> dict[str, float | int]:
    signed_error = np.array([r["signed_error_min"] for r in rows])
    absolute_error = np.abs(signed_error)
    delta = absolute_error - original_error
    confusion = sum(row["confusion"] for row in rows)
    correct = np.diag(confusion)
    predicted, actual = confusion.sum(axis=0), confusion.sum(axis=1)
    denom = predicted + actual
    f1 = np.divide(2 * correct, denom, out=np.zeros(5), where=denom > 0)
    return {
        "mae_min": float(absolute_error.mean()),
        "mean_signed_error_min": float(signed_error.mean()),
        "improved_nights": int(np.count_nonzero(delta < 0)),
        "unchanged_nights": int(np.count_nonzero(delta == 0)),
        "increased_error_nights": int(np.count_nonzero(delta > 0)),
        "error_increase_ge10_nights": int(np.count_nonzero(delta >= 10)),
        "error_increase_ge10_pct": float(100 * np.mean(delta >= 10)),
        "absolute_error_le10_nights": int(np.count_nonzero(absolute_error <= 10)),
        "absolute_error_le10_pct": float(100 * np.mean(absolute_error <= 10)),
        "n1_f1_pct": float(100 * f1[1]),
        "macro_f1_pct": float(100 * f1.mean()),
        "n1_precision_pct": float(100 * correct[1] / predicted[1]) if predicted[1] else 0.0,
        "n1_recall_pct": float(100 * correct[1] / actual[1]) if actual[1] else 0.0,
        "edited_epochs": sum(r["edits"] for r in rows),
    }


def paired_ci(values: np.ndarray, seed: int, repetitions: int) -> list[float]:
    indices = np.random.default_rng(seed).integers(0, len(values), size=(repetitions, len(values)))
    return np.quantile(values[indices].mean(axis=1), (0.025, 0.975)).tolist()


def evaluate(
    initial_path: Path, expanded_path: Path | None, evaluation_path: Path, settings: dict[str, int],
) -> dict[str, Any]:
    initial, expanded, evaluation = load_cohorts(initial_path, expanded_path, evaluation_path)
    decision = select_with_expansion(initial, expanded or None, settings["seed20"], settings["seed40"])
    calibration = expanded if decision["expanded"] else initial
    biases = fit_biases(calibration)
    labels_sha256 = hashlib.sha256()
    scores = {base: {name: [] for name in METHODS} for base in BASES}
    records = []
    entries = json.loads(evaluation_path.read_text(encoding="utf-8"))
    for entry, pair in zip(entries, evaluation, strict=True):
        full = apply_policy(pair, decision["policy"])
        records.append(pair[BASES[0]].record)
        for base, night in pair.items():
            biased = n1_bias.decode(night.probabilities, biases[base]["Unconstrained Bias"])
            aligned = aligned_bias.apply_aligned(as_row(night), biases[base]["Aligned Bias"], full[base])
            # Read evaluation labels only after decoding.
            with np.load(entry[base], allow_pickle=False) as source:
                reference = source["y_true"]
            if reference.shape != night.original.shape or not np.isin(reference, range(5)).all():
                raise ValueError(f"Reference labels do not match the prediction epochs for {night.record}.")
            for method, labels in zip(METHODS, (night.original, full[base], biased, aligned), strict=True):
                labels_sha256.update(f"{night.record}/{base}/{method}/".encode())
                labels_sha256.update(labels.astype(np.uint8).tobytes())
                scores[base][method].append(score_night(labels, reference, night.original))

    models = {}
    for base, methods in scores.items():
        errors = {method: np.abs([r["signed_error_min"] for r in rows]) for method, rows in methods.items()}
        original = errors["Original"]
        gain = original - errors["Full"]
        bias_increase = errors["Unconstrained Bias"] - original >= 10
        full_increase = gain <= -10
        rate_gap = 100 * (bias_increase.astype(int) - full_increase)
        models[base] = {
            "methods": {name: summarize(rows, original) for name, rows in methods.items()},
            "full_mae_reduction_min": float(gain.mean()),
            "full_mae_reduction_ci95_min": paired_ci(gain, settings["primary_bootstrap_seed"], settings["bootstrap_draws"]),
            "bias_minus_full_ge10_pp": float(rate_gap.mean()),
            "bias_minus_full_ge10_ci95_pp": paired_ci(
                rate_gap[np.argsort(records)], settings["comparison_bootstrap_seed"], settings["bootstrap_draws"],
            ),
        }
    return {
        "calibration_nights": len(calibration), "evaluation_nights": len(evaluation),
        "decision": decision, "biases": biases, "models": models,
        "labels_sha256": labels_sha256.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    demo = modes.add_parser("demo", help="Run the synthetic example")
    demo.add_argument("--output", type=Path, default=Path("demo_output.json"),
                      help="JSON output path (default: demo_output.json)")
    actual = modes.add_parser("evaluate", help="Evaluate frozen predictions")
    actual.add_argument("--calibration20", type=Path, required=True,
                        help="Manifest for the initial 20 calibration subjects")
    actual.add_argument("--calibration40", type=Path,
                        help="Optional manifest for 21 to 40 calibration subjects, including the initial 20")
    actual.add_argument("--evaluation", type=Path, required=True,
                        help="Manifest with one evaluation night per subject")
    actual.add_argument("--output", type=Path, default=Path("results.json"),
                        help="JSON output path (default: results.json)")
    args = parser.parse_args()
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.mode == "demo":
        from demo import run_demo

        result = run_demo(evaluate, settings)
        result["input"] = "synthetic example"
    else:
        result = evaluate(args.calibration20, args.calibration40, args.evaluation, settings)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{result['decision']['policy']} | {result['evaluation_nights']} evaluation nights")
    for base, model in result["models"].items():
        print(base)
        for name, row in model["methods"].items():
            print(f"  {name}: MAE {row['mae_min']:.2f} min. "
                  f"Nights with absolute error increase >=10 min: {row['error_increase_ge10_nights']}")
    print(args.output)


if __name__ == "__main__":
    main()
