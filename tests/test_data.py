import numpy as np
from mlops.data import generate_dataset, validate_dataframe, build_reference_profile


def test_generator_is_deterministic():
    a = generate_dataset(500, "baseline", seed=1); b = generate_dataset(500, "baseline", seed=1)
    assert a.equals(b)


def test_scenarios_actually_drift():
    base = generate_dataset(5000, "baseline", seed=1)
    sev = generate_dataset(5000, "severe_drift", seed=1)
    assert sev["monthly_charges"].mean() > base["monthly_charges"].mean() * 1.2
    assert sev["age"].mean() < base["age"].mean() - 5


def test_concept_drift_changes_label_not_features():
    base = generate_dataset(20000, "baseline", seed=1); cd = generate_dataset(20000, "concept_drift", seed=1)
    assert abs(base["age"].mean() - cd["age"].mean()) < 0.5
    assert cd["churned"].mean() > base["churned"].mean() + 0.02


def test_clean_data_passes_validation(cfg):
    assert validate_dataframe(generate_dataset(3000, "baseline"), cfg).passed


def test_validation_catches_problems(cfg):
    df = generate_dataset(3000, "baseline")
    df.loc[:10, "age"] = 500
    df.loc[:50, "contract_type"] = "lifetime"
    df["tenure_months"] = np.nan
    rep = validate_dataframe(df, cfg)
    names = {(c.name, c.column) for c in rep.failures}
    assert not rep.passed
    assert ("value_range", "age") in names and ("allowed_categories", "contract_type") in names
    assert ("null_fraction", "tenure_months") in names


def test_validation_flags_too_few_rows(cfg):
    assert not validate_dataframe(generate_dataset(100, "baseline"), cfg).passed


def test_reference_profile_shape(cfg):
    prof = build_reference_profile(generate_dataset(3000, "baseline"), cfg)
    assert set(prof["numeric"]) == set(cfg["schema"]["numeric"])
    assert abs(sum(prof["numeric"]["age"]["bin_probs"]) - 1) < 1e-9
    assert abs(sum(prof["categorical"]["contract_type"]["freqs"].values()) - 1) < 1e-9
