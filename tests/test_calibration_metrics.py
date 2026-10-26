import numpy as np

from iemocap_mer.metrics import (
    apply_temperature,
    bootstrap_calibration_cis,
    calibration_report,
    equal_mass_ece_mce,
    equal_width_ece_mce,
    fit_temperature,
    multiclass_brier,
    multiclass_nll,
)


def test_nll_one_hot_and_uniform() -> None:
    labels = np.array([0, 1, 2, 3])
    one_hot = np.eye(4)
    nll = multiclass_nll(one_hot, labels)
    assert nll < 1e-8

    uniform = np.full((4, 4), 0.25)
    expected = -np.log(0.25)
    assert abs(multiclass_nll(uniform, labels) - expected) < 1e-12


def test_brier_one_hot_and_uniform() -> None:
    labels = np.array([0, 1, 2, 3])
    one_hot = np.eye(4)
    assert multiclass_brier(one_hot, labels) < 1e-12

    uniform = np.full((4, 4), 0.25)
    # (0.25-1)^2 + 3*(0.25)^2 = 0.75
    assert abs(multiclass_brier(uniform, labels) - 0.75) < 1e-12


def test_equal_width_ece_perfect_and_gap() -> None:
    n = 200
    conf = np.full(n, 0.9)
    perfect = np.concatenate([np.ones(180), np.zeros(20)])
    ece, mce, _ = equal_width_ece_mce(conf, perfect, n_bins=15)
    assert ece < 1e-12
    assert mce < 1e-12

    always_wrong = np.zeros(n)
    ece_bad, mce_bad, _ = equal_width_ece_mce(conf, always_wrong, n_bins=15)
    assert abs(ece_bad - 0.9) < 1e-12
    assert abs(mce_bad - 0.9) < 1e-12


def test_equal_mass_ece_quantile_bins() -> None:
    # Skewed confidences: 10 low / 90 high. Equal-mass (M=2) splits by rank
    # (50/50) and therefore mixes the two groups; equal-width keeps them apart.
    conf = np.concatenate([np.full(10, 0.10), np.full(90, 0.95)])
    correct = np.concatenate([np.ones(5), np.zeros(5), np.ones(81), np.zeros(9)])
    ece_m, _, rows = equal_mass_ece_mce(conf, correct, n_bins=2)
    # After ranking: bin 0 = 10 low + 40 high (acc 45/50, conf 0.78);
    # bin 1 = remaining 50 high (acc 41/50, conf 0.95).
    expected_mass = 0.5 * abs(0.90 - 0.78) + 0.5 * abs(0.82 - 0.95)
    assert abs(ece_m - expected_mass) < 1e-12
    assert [r["bin_count"] for r in rows] == [50, 50]

    ece_w, _, _ = equal_width_ece_mce(conf, correct, n_bins=15)
    expected_width = 0.1 * abs(0.50 - 0.10) + 0.9 * abs(0.90 - 0.95)
    assert abs(ece_w - expected_width) < 1e-12
    assert abs(ece_w - ece_m) > 1e-6


def test_temperature_scaling_recovers_overconfidence() -> None:
    rng = np.random.default_rng(0)
    n, k = 800, 4
    true_logits = rng.normal(size=(n, k))
    true_probs = apply_temperature(true_logits, 1.0)
    labels = np.array([rng.choice(k, p=p) for p in true_probs])
    # Sharpen logits relative to the generating distribution (overconfidence).
    # NLL-optimal T should recover the sharpening factor.
    sharp = 4.0 * true_logits
    t_hat = fit_temperature(sharp, labels)
    assert 2.5 < t_hat < 6.0
    nll_raw = multiclass_nll(apply_temperature(sharp, 1.0), labels)
    nll_cal = multiclass_nll(apply_temperature(sharp, t_hat), labels)
    assert nll_cal < nll_raw
    t_id = fit_temperature(true_logits, labels)
    assert 0.6 < t_id < 1.6


def test_calibration_report_and_bootstrap_keys() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(80, 4))
    probs = apply_temperature(logits, 1.0)
    labels = rng.integers(0, 4, size=80)
    report = calibration_report(labels, probs, n_bins=15)
    assert report["n_samples"] == 80
    for key in ("accuracy", "nll", "brier", "ece_width", "ece_mass", "mce"):
        assert np.isfinite(report[key])

    cis = bootstrap_calibration_cis(
        labels, probs, n_resamples=50, seed=42, n_bins=15
    )
    for key in ("accuracy", "nll", "brier", "ece_width", "ece_mass", "mce"):
        assert cis[key]["low"] <= cis[key]["point"] <= cis[key]["high"] or np.isclose(
            cis[key]["low"], cis[key]["point"]
        )
