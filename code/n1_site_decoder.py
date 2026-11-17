"""Select a site rule on calibration nights and apply it to frozen predictions."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BASES = ("usleep", "deepresnet")
VERSION = "r235-stage7-20260929"
POLICIES = ("ratio_only", "consensus_half24", "threshold_0.2_cap48",
            "threshold_0.3_cap48")


@dataclass(frozen=True)
class Night:
    record: str
    epoch_indices: np.ndarray
    probabilities: np.ndarray
    original: np.ndarray
    reference: np.ndarray | None = None

    def validate(self) -> None:
        n = len(self.original)
        if n == 0 or self.probabilities.shape != (n, 5) or len(self.epoch_indices) != n:
            raise ValueError(f"Invalid array lengths for {self.record}")
        if self.reference is not None and len(self.reference) != n:
            raise ValueError(f"Reference length mismatch for {self.record}")
        if np.any(np.diff(self.epoch_indices) <= 0):
            raise ValueError(f"Epoch indices must be strictly increasing: {self.record}")
        if not np.isfinite(self.probabilities).all() or \
           np.any(self.probabilities < 0) or np.any(self.probabilities > 1):
            raise ValueError(f"Invalid probability values: {self.record}")
        if not np.isin(self.original, np.arange(5)).all():
            raise ValueError(f"Invalid original stages: {self.record}")
        if self.reference is not None and not np.isin(self.reference, np.arange(5)).all():
            raise ValueError(f"Invalid reference stages: {self.record}")


def read_night(path: str | Path, *, with_reference: bool) -> Night:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        required = ("epoch_indices", "probabilities", "original")
        if any(key not in z for key in required):
            raise ValueError(f"Missing required array in {path}")
        if with_reference and "y_true" not in z:
            raise ValueError(f"Calibration reference is missing: {path}")
        night = Night(path.stem, z["epoch_indices"], z["probabilities"],
                      z["original"], z["y_true"] if with_reference else None)
    night.validate()
    return night


def threshold(night: Night, value: float, cap: int = 48) -> np.ndarray:
    original, p = night.original, night.probabilities
    output = original.copy()
    add = np.flatnonzero(np.isin(original, (0, 2)) & (p[:, 1] >= value))
    remove = np.flatnonzero((original == 1) & (p[:, 1] < value))
    add = add[np.argsort(-p[add, 1], kind="stable")[:cap]]
    remove = remove[np.argsort(p[remove, 1], kind="stable")[:cap]]
    output[add] = 1
    output[remove] = np.where(p[remove, 0] >= p[remove, 2], 0, 2)
    return output


def ratio_only(night: Night) -> np.ndarray:
    original, p = night.original, night.probabilities
    hard = int(np.sum(original == 1))
    soft = float(np.sum(p[:, 1]))
    if hard / max(soft, 1e-7) >= .55:
        return original.copy()
    target = hard + .5 * abs(soft - hard)
    delta = round(float(target)) - hard
    output = original.copy()
    if delta > 0:
        eligible = np.flatnonzero(np.isin(original, (0, 2)) & (p[:, 1] >= .05))
        selected = eligible[np.argsort(-p[eligible, 1], kind="stable")[:delta]]
        output[selected] = 1
    return output


def signed_decode(night: Night, fraction_shift: float,
                  scale: float = .5, cap: int = 24) -> np.ndarray:
    original, p = night.original, night.probabilities
    output = original.copy()
    amount = min(cap, round(scale * abs(fraction_shift) * len(original)))
    if amount == 0:
        return output
    if fraction_shift > 0:
        allowed = np.isin(original, (0, 2)) & (p[:, 1] >= .05)
        candidate = np.flatnonzero(allowed)
        margin = p[candidate, 1] - np.maximum(p[candidate, 0], p[candidate, 2])
        selected = candidate[np.argsort(-margin, kind="stable")[:amount]]
        output[selected] = 1
    else:
        candidate = np.flatnonzero(original == 1)
        margin = p[candidate, 1] - np.maximum(p[candidate, 0], p[candidate, 2])
        selected = candidate[np.argsort(margin, kind="stable")[:amount]]
        output[selected] = np.where(p[selected, 0] >= p[selected, 2], 0, 2)
    return output


def consensus(pair: dict[str, Night]) -> dict[str, np.ndarray]:
    first, second = pair[BASES[0]], pair[BASES[1]]
    result = {}
    for base, own, peer in ((BASES[0], first, second),
                            (BASES[1], second, first)):
        common, own_pos, peer_pos = np.intersect1d(
            own.epoch_indices, peer.epoch_indices, return_indices=True)
        if len(common) < .75 * min(len(own.epoch_indices), len(peer.epoch_indices)):
            raise ValueError(f"Insufficient two-base epoch overlap: {own.record}")
        own_soft = own.probabilities[:, 1].astype(float)
        peer_on_own = own_soft.copy()
        peer_on_own[own_pos] = peer.probabilities[peer_pos, 1]
        target = .5 * (np.mean(own_soft) + np.mean(peer_on_own))
        shift = target - np.mean(own.original == 1)
        result[base] = signed_decode(own, float(shift), .5, 24)
    return result


def candidates(pair: dict[str, Night]) -> dict[str, dict[str, np.ndarray]]:
    if set(pair) != set(BASES) or pair[BASES[0]].record != pair[BASES[1]].record:
        raise ValueError("A pair needs matching record names for both bases")
    return {
        "ratio_only": {base: ratio_only(pair[base]) for base in BASES},
        "consensus_half24": consensus(pair),
        "threshold_0.2_cap48": {base: threshold(pair[base], .2) for base in BASES},
        "threshold_0.3_cap48": {base: threshold(pair[base], .3) for base in BASES},
    }


def gain(night: Night, output: np.ndarray) -> float:
    if night.reference is None:
        raise ValueError(f"Calibration reference required: {night.record}")
    truth = int(np.sum(night.reference == 1))
    return .5 * (abs(int(np.sum(night.original == 1)) - truth) -
                 abs(int(np.sum(output == 1)) - truth))


def select_policy(calibration: list[dict[str, Night]], seed: int) -> tuple[str, dict]:
    if not calibration:
        raise ValueError("Calibration nights are required")
    by_policy = {name: {base: [] for base in BASES} for name in POLICIES}
    for pair in calibration:
        menu = candidates(pair)
        for name in POLICIES:
            for base in BASES:
                by_policy[name][base].append(gain(pair[base], menu[name][base]))
    diagnostics = {}
    eligible = []
    for name in POLICIES:
        details = {}
        for base_i, base in enumerate(BASES):
            values = np.asarray(by_policy[name][base], dtype=float)
            rng = np.random.default_rng(seed + len(name) + base_i)
            draws = rng.integers(0, len(values), size=(3000, len(values)))
            lower = float(np.quantile(values[draws].mean(axis=1), .025))
            details[base] = {"mean_gain_min": float(values.mean()),
                             "lower95_gain_min": lower}
        diagnostics[name] = details
        if all(details[base]["mean_gain_min"] > .5 and
               details[base]["lower95_gain_min"] > 0 for base in BASES):
            eligible.append((min(details[base]["lower95_gain_min"]
                                 for base in BASES), name))
    return (max(eligible)[1] if eligible else "raw"), diagnostics


def select_with_expansion(initial: list[dict[str, Night]],
                          expanded: list[dict[str, Night]] | None,
                          seed20: int, seed40: int) -> dict:
    first, first_diag = select_policy(initial, seed20)
    if first != "raw":
        return {"policy": first, "initial_policy": first,
                "calibration_nights": len(initial), "expanded": False,
                "diagnostics": first_diag, "abstention_reason": None}
    if expanded is None or len(expanded) <= len(initial):
        return {"policy": "raw", "initial_policy": "raw",
                "calibration_nights": len(initial), "expanded": False,
                "diagnostics": first_diag,
                "abstention_reason": "no_candidate_passed_with_available_calibration"}
    second, second_diag = select_policy(expanded, seed40)
    return {"policy": second, "initial_policy": first,
            "calibration_nights": len(expanded), "expanded": True,
            "diagnostics": second_diag,
            "abstention_reason": None if second != "raw" else
                                "no_candidate_passed_after_expansion"}


def apply_policy(pair: dict[str, Night], policy: str) -> dict[str, np.ndarray]:
    if policy == "raw":
        return {base: pair[base].original.copy() for base in BASES}
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    return candidates(pair)[policy]


def _read_manifest(path: Path, *, with_reference: bool) -> tuple[list[dict[str, Night]], set[str]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("Manifest must contain an array of paired nights")
    pairs = []
    subjects: set[str] = set()
    records: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("subject_id"), str) or not entry["subject_id"]:
            raise ValueError(f"Each manifest entry needs a nonempty subject_id: {path}")
        pair = {base: read_night(entry[base], with_reference=with_reference)
                for base in BASES}
        if pair[BASES[0]].record != pair[BASES[1]].record:
            raise ValueError(f"Base records differ in {path}")
        record = pair[BASES[0]].record
        if record in records:
            raise ValueError(f"Duplicate record in {path}: {record}")
        records.add(record)
        subjects.add(entry["subject_id"])
        pairs.append(pair)
    return pairs, subjects


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-site-calibrated constrained N1 decoding")
    parser.add_argument("--calibration20", type=Path, required=True)
    parser.add_argument("--calibration40", type=Path)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed20", type=int, required=True)
    parser.add_argument("--seed40", type=int, required=True)
    args = parser.parse_args()
    initial, initial_subjects = _read_manifest(args.calibration20, with_reference=True)
    if len(initial_subjects) != 20:
        raise ValueError("calibration20 must contain exactly 20 distinct subjects")
    if args.calibration40:
        expanded, expanded_subjects = _read_manifest(args.calibration40, with_reference=True)
        if not initial_subjects.issubset(expanded_subjects) or not (20 < len(expanded_subjects) <= 40):
            raise ValueError("calibration40 must extend the first 20 to 21–40 subjects")
        initial_records = {pair[BASES[0]].record for pair in initial}
        expanded_records = {pair[BASES[0]].record for pair in expanded}
        if not initial_records.issubset(expanded_records):
            raise ValueError("calibration40 must retain every calibration20 night")
    else:
        expanded, expanded_subjects = None, set()
    test, test_subjects = _read_manifest(args.test, with_reference=False)
    if (initial_subjects | expanded_subjects) & test_subjects:
        raise ValueError("Calibration and test subjects overlap")
    if set(x[BASES[0]].record for x in initial) & set(x[BASES[0]].record for x in test):
        raise ValueError("Calibration and test records overlap")
    decision = select_with_expansion(initial, expanded, args.seed20, args.seed40)
    args.output.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, Any]] = []
    for pair in test:
        corrected = apply_policy(pair, decision["policy"])
        for base in BASES:
            night = pair[base]
            changed = corrected[base] != night.original
            destination = args.output / f"{night.record}.{base}.npz"
            np.savez_compressed(destination, epoch_indices=night.epoch_indices,
                                original=night.original, corrected=corrected[base],
                                changed_mask=changed)
            log.append({"record": night.record, "base": base,
                        "modified_epochs": int(changed.sum()),
                        "output": str(destination)})
    (args.output / "decision.json").write_text(
        json.dumps({"software_version": VERSION,
                    "calibration_seeds": {"seed20": args.seed20, "seed40": args.seed40},
                    "decision": decision, "outputs": log}, ensure_ascii=False,
                   indent=2), encoding="utf-8")
    print(json.dumps({"policy": decision["policy"], "expanded": decision["expanded"],
                      "abstention_reason": decision["abstention_reason"],
                      "test_nights": len(test)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
