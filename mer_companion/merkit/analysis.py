from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from scipy import stats
from .common import *

PROBS = [f"p_{k}" for k in range(4)]
LOGITS = [f"z_{k}" for k in range(4)]
KEY = ["model", "seed", "fold", "utterance_id", "condition"]

def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 4 or not np.isfinite(z).all():
        raise ValueError("Expected finite Nx4 logits")
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))

def validate_table(frame, split):
    required = set(KEY + ["split", "label"] + PROBS + LOGITS)
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing columns or empty prediction table")
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Null or duplicate prediction identity")
    if set(frame["split"]) != {split}:
        raise ValueError(f"Expected split={split}")
    y = frame.label.to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isin(y, range(4)).all():
        raise ValueError("Labels must be integers 0..3")
    p = frame[PROBS].to_numpy(dtype=float)
    z = frame[LOGITS].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid probability value")
    if not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError("Probability rows must sum to one")
    if not np.allclose(p, softmax(z), atol=1e-6, rtol=1e-6):
        raise ValueError("Probabilities disagree with original logits")
    if not np.array_equal(p.argmax(1), z.argmax(1)):
        raise ValueError("Probability/logit argmax mismatch")
    return y.astype(int), p, z

def classification(y, predicted):
    y, predicted = np.asarray(y, dtype=int), np.asarray(predicted, dtype=int)
    if len(y) == 0 or y.shape != predicted.shape:
        raise ValueError("Invalid classification arrays")
    cm = np.bincount(4 * y + predicted, minlength=16).reshape(4, 4)
    tp = cm.diagonal().astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(4), where=denom > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(4), where=cm.sum(1) > 0)
    return {"macro_f1": float(f1.mean()), "uar": float(recall.mean()),
            "accuracy": float((y == predicted).mean())}

def score(y, z, temperature=1.0, bins=15):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    effective = np.asarray(z, dtype=float) / temperature
    p = softmax(effective)
    y = np.asarray(y, dtype=int)
    pred, conf = p.argmax(1), p.max(1)
    correct = pred == y
    index = np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1])
    width = [np.flatnonzero(index == b) for b in range(bins)]
    mass = np.array_split(np.argsort(conf, kind="stable"), min(bins, len(y)))
    def gaps(groups):
        return [(len(g), abs(float(correct[g].mean() - conf[g].mean()))) for g in groups if len(g)]
    wg, mg = gaps(width), gaps(mass)
    return {"n": int(len(y)), **classification(y, pred),
        "nll": float((logsumexp(effective, axis=1) - effective[np.arange(len(y)), y]).mean()),
        "brier": float(((p - np.eye(4)[y]) ** 2).sum(1).mean()),
        "ece_width": sum(n * g for n, g in wg) / len(y),
        "ece_mass": sum(n * g for n, g in mg) / len(y), "mce_width": max(g for _, g in wg),
        "mean_confidence": float(conf.mean()), "confidence_minus_accuracy": float(conf.mean() - correct.mean())}

def fit_temperature(y, z):
    y, z = np.asarray(y, dtype=int), np.asarray(z, dtype=float)
    softmax(z)
    lo, hi = np.log(.05), np.log(20.)
    def nll(logt):
        v = z / np.exp(logt)
        return float((logsumexp(v, axis=1) - v[np.arange(len(y)), y]).mean())
    fitted = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-8})
    if not fitted.success or not np.isfinite(fitted.fun):
        raise ValueError("Temperature fitting failed")
    # Include endpoints and T=1; all choices use development data only.
    logt = min([lo, hi, 0., float(fitted.x)], key=nll)
    return float(np.exp(logt)), {"validation_nll_before": nll(0.),
        "validation_nll_after": nll(logt), "boundary": min(logt - lo, hi - logt) < 1e-4}

def paired(clean, changed, temperature=1.0):
    for f in (clean, changed):
        validate_table(f, "test")
        if f.utterance_id.duplicated().any():
            raise ValueError("Duplicate IDs within a run/condition")
    a, b = clean.set_index("utterance_id"), changed.set_index("utterance_id")
    if set(a.index) != set(b.index):
        raise ValueError("Clean and altered ID sets differ")
    b = b.loc[a.index]
    if not np.array_equal(a.label, b.label):
        raise ValueError("Labels disagree after pairing")
    pa = softmax(a[LOGITS].to_numpy() / temperature)
    pb = softmax(b[LOGITS].to_numpy() / temperature)
    y = a.label.to_numpy(dtype=int)
    ca, cb = pa.max(1), pb.max(1)
    correct_a, correct_b = pa.argmax(1) == y, pb.argmax(1) == y
    result = {"n": len(y), "vrr": float((cb > ca).mean()),
        "mean_clean_minus_altered_confidence": float((ca - cb).mean()),
        "clean_correct_altered_wrong_fraction": float((correct_a & ~correct_b).mean()),
        "clean_wrong_altered_correct_fraction": float((~correct_a & correct_b).mean()),
        "prediction_change_fraction": float((pa.argmax(1) != pb.argmax(1)).mean())}
    ac = classification(y, pa.argmax(1)); bc = classification(y, pb.argmax(1))
    result.update(macro_f1_drop=ac["macro_f1"] - bc["macro_f1"], uar_drop=ac["uar"] - bc["uar"])
    return result

def analyze_frames(dev, test, expected_conditions):
    yd, _, zd = validate_table(dev, "validation")
    yt, _, zt = validate_table(test, "test")
    if set(dev.condition) != {"clean"}:
        raise ValueError("Temperature must be fitted on clean development rows only")
    for frame in (dev, test):
        if len(frame[["model", "seed", "fold"]].drop_duplicates()) != 1:
            raise ValueError("Each export file must contain exactly one run")
    if tuple(dev.iloc[0][["model", "seed", "fold"]]) != tuple(test.iloc[0][["model", "seed", "fold"]]):
        raise ValueError("Development/test run identity mismatch")
    if set(dev.utterance_id) & set(test.utterance_id):
        raise ValueError("Development/test utterance leakage")
    if set(test.condition) != set(expected_conditions) or "clean" not in expected_conditions:
        raise ValueError("Condition coverage differs from export manifest")
    temperature, info = fit_temperature(yd, zd)
    if not np.array_equal(softmax(zt).argmax(1), softmax(zt / temperature).argmax(1)):
        raise ValueError("Numerical argmax change after temperature scaling")
    scores, pairs = [], []
    clean = test[test.condition == "clean"]
    for condition, group in test.groupby("condition", sort=True):
        y, _, z = validate_table(group, "test")
        if set(group.utterance_id) != set(clean.utterance_id):
            raise ValueError("Condition ID coverage mismatch")
        for calibration, value in (("raw", 1.0), ("temperature", temperature)):
            scores.append(dict(condition=condition, calibration=calibration, **score(y, z, value)))
            if condition != "clean":
                pairs.append(dict(condition=condition, calibration=calibration, **paired(clean, group, value)))
    return scores, pairs, {"temperature": temperature, "n_validation": len(dev), **info}

def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    missing = ~np.isfinite(p)
    safe = np.where(missing, 1., p)
    order = np.argsort(safe, kind="stable")
    corrected = np.minimum(1., np.maximum.accumulate((len(p) - np.arange(len(p))) * safe[order]))
    out = np.empty(len(p)); out[order] = corrected; out[missing] = np.nan
    return out

def primary_tests(scores):
    clean = scores[(scores.condition == "clean") & (scores.calibration == "raw")]
    expected = {(s, f) for s in SEEDS for f in FOLDS}
    for model in SPECS:
        part = clean[clean.model == model]
        keys = set(zip(part.seed, part.fold))
        if keys != expected or len(part) != 15:
            return [], [], "Primary tests not computed: complete 5-model x 3-seed x 5-fold coverage is required."
    means = clean.groupby(["model", "fold"]).macro_f1.mean()
    rows, differences = [], []
    for index, (a, b) in enumerate(PRIMARY_PAIRS, 1):
        d = np.array([means[b, fold] - means[a, fold] for fold in FOLDS])
        mean, sd = float(d.mean()), float(d.std(ddof=1))
        if sd > 0:
            t_value = mean / (sd / np.sqrt(5))
            p = float(2 * stats.t.sf(abs(t_value), 4))
            half = float(stats.t.ppf(.975, 4) * sd / np.sqrt(5))
            note = "paired t; exploratory under small and overlapping CV folds"
        else:
            p = 1. if mean == 0 else np.nan
            half = 0.; note = "zero variance in paired differences; nonzero constant effect has undefined t-test"
        rows.append(dict(comparison=index, model_a=a, model_b=b, effect_b_minus_a=mean,
                         n_session_folds=5, raw_p=p, ci95_t_low=mean-half, ci95_t_high=mean+half,
                         family="historical_seven_posthoc", note=note))
        differences.extend(dict(comparison=index, model_a=a, model_b=b, fold=fold,
            difference_b_minus_a=float(d[j])) for j, fold in enumerate(FOLDS))
    adjusted = holm([r["raw_p"] for r in rows])
    for row, p in zip(rows, adjusted):
        row["holm_p"] = float(p)
    return rows, differences, "Historical family is post hoc; test assumptions and n=5 limit interpretation."

def run_analysis(args):
    exports = Path(args.exports).resolve()
    manifest = read_json(exports / "export_manifest.json")
    if not manifest["complete"]:
        raise ValueError("Export has failed/incomplete runs; resolve and resume before analysis")
    if manifest["selected_count"] != 75 and not args.allow_partial:
        raise ValueError("This is a partial/smoke export. Use --allow-partial for diagnostics only.")
    if len(manifest["runs"]) != manifest["selected_count"] or \
            {r["run_id"] for r in manifest["runs"]} != set(manifest["settings"]["selected"]):
        raise ValueError("Selected and completed run identities differ")
    if sha256(exports / "split_manifest.csv") != manifest["settings"]["split_manifest_sha256"]:
        raise ValueError("Split manifest changed after export")
    out = new_output(args.out)
    score_rows, pair_rows, temperatures, input_hashes = [], [], [], {}
    expected_split = pd.read_csv(exports / "split_manifest.csv", dtype={"utterance_id": str})
    seen = set()
    for run in manifest["runs"]:
        key = (run["model"], run["seed"], run["fold"])
        if key in seen:
            raise ValueError("Duplicate run in export manifest")
        seen.add(key)
        frames = {}
        for part in ("validation", "test"):
            record = run["files"][part]
            relative = "runs/" + run["run_id"] + "/" + record["file"]
            path = within(exports, relative)
            if sha256(path) != record["sha256"]:
                raise ValueError("Prediction file changed since export")
            input_hashes[relative] = record["sha256"]
            frame = pd.read_csv(path, dtype={"utterance_id": str})
            subset = expected_split[(expected_split.fold == run["fold"]) & (expected_split.split == part)]
            truth = subset.set_index("utterance_id").label.to_dict()
            if set(frame.utterance_id) != set(truth):
                raise ValueError("Prediction IDs disagree with the saved split manifest")
            if any(truth[u] != y for u, y in zip(frame.utterance_id, frame.label)):
                raise ValueError("Prediction labels disagree with the split manifest")
            if len(frame) != record["rows"]:
                raise ValueError("Prediction row count disagrees with export manifest")
            if any(set(frame[k]) != {run[k]} for k in ("model", "seed", "fold")):
                raise ValueError("Prediction run identity disagrees with export manifest")
            frames[part] = frame
        scores, pairs, temp = analyze_frames(frames["validation"], frames["test"], run["conditions"])
        meta = dict(dataset="iemocap", model=run["model"], seed=run["seed"], fold=run["fold"],
                    history_status=run["history_status"])
        score_rows.extend({**meta, **x} for x in scores)
        pair_rows.extend({**meta, **x} for x in pairs)
        temperatures.append({**meta, **temp})
    scores = pd.DataFrame(score_rows)
    scores.to_csv(out / "fold_scores.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "paired_conditions.csv", index=False)
    pd.DataFrame(temperatures).to_csv(out / "temperatures.csv", index=False)
    metric_names = ["macro_f1", "uar", "accuracy", "nll", "brier", "ece_width", "ece_mass", "mce_width"]
    summaries = []
    for keys, group in scores.groupby(["model", "condition", "calibration"], sort=True):
        row = dict(zip(("model", "condition", "calibration"), keys))
        row.update(n_runs=len(group), complete_15_run_coverage=
            set(zip(group.seed, group.fold)) == {(s, f) for s in SEEDS for f in FOLDS})
        for metric in metric_names:
            per_seed = group.groupby("seed")[metric].mean()
            row[metric + "_mean_of_seed_means"] = float(per_seed.mean())
            row[metric + "_sd_seed_means_ddof0"] = float(per_seed.std(ddof=0))
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out / "summary_scores.csv", index=False)
    comparisons, differences, note = primary_tests(scores)
    if comparisons:
        pd.DataFrame(comparisons).to_csv(out / "comparison_tests.csv", index=False)
        pd.DataFrame(differences).to_csv(out / "paired_fold_differences.csv", index=False)
    analysis = {"created": timestamp(), "dataset": "iemocap", "export_manifest_sha256": sha256(exports / "export_manifest.json"),
        "toolkit_hashes": toolkit_hashes(),
        "historical_review_required": manifest["history_unknown_count"] > 0,
        "complete_principal_coverage": len(seen) == 75,
        "prediction_hashes": input_hashes, "n_runs": len(seen), "history_unknown_count": manifest["history_unknown_count"],
        "training_performed": False, "temperature_bounds": [.05, 20.], "temperature_fit_split": "clean development only",
        "test_labels_used_to_fit_temperature": False, "ece_bins": 15, "vrr_rule": "strict max(p_altered) > max(p_clean); exact ties not violations",
        "summary_sd_ddof": 0, "primary_test_note": note,
        "limitations": ["Unknown historical training fields remain unresolved after inference.",
            "No second-dataset training, new architecture training, or fully-masked model path is implemented.",
            "No claim of independent 15-run or utterance-level statistical samples.",
            "Calibration tables are per-run and descriptive summaries; no pooled utterance bootstrap CI is added.",
            "t intervals use five session differences and do not remove overlapping-training dependence.",
            "Original logits plus temperatures reproduce scaled predictions; a redundant scaled CSV is not written."]}
    write_json(out / "analysis_manifest.json", analysis)
    print(f"Analysis saved: {out} ({len(seen)} runs)", flush=True)
    return 0
