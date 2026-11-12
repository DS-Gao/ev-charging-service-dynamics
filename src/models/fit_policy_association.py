"""Fit the six policy-background service-conversion models.

The public entry point reads the column-reduced, eligible policy-model input
under ``data/processed``. The same complete sample and design matrix are used
for the three outcomes (N, A and I) under the base and COVID-adjusted
specifications. The adjusted specification adds the observed-date Oxford
stringency exposure divided by 10. GDP, area, jackknife, and bootstrap
estimates are not part of this estimator.

The estimator is a QR OLS solve with province-clustered CR1 covariance,
using ``Gp/(Gp-1) * (n-1)/(n-K)`` and t(Gp-1) intervals.  ``K`` is the full
rank of the displayed intercept + observed covariates + city/year FE design.
The fit cache signature contains only actual X, Y, province cluster arrays and
the estimation-math version; prose/metadata changes therefore do not trigger
a fit.  A small synthetic covariance check runs before any real solve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve()
REPOSITORY = HERE.parents[2]
OUTPUT = REPOSITORY / "data" / "results" / "recomputed_policy"
OUTPUT.mkdir(parents=True, exist_ok=True)
PUBLIC_INPUT = REPOSITORY / "data" / "processed" / "policy_model_input.csv"

OUTCOMES = ("N", "A", "I")
SPECS = ("C0", "C1")
POLICY = ("R_prev_observed_v9", "S_prev_observed_v9")
WEATHER = (
    "weather_temperature",
    "weather_precipitation",
    "weather_wind_speed",
    "weather_relative_humidity",
)
BASE_CONTROLS = ("lag1_B_count_later", *WEATHER)
COVID_RAW = "covid_stringency_source_date_mean"
COVID_SCALED = "covid_stringency_source_date_mean_per10"
MATH_VERSION = "v11_qr_ols_province_cr1_2026-11-03"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(dict(payload)), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY.resolve()).as_posix()
    except ValueError:
        return path.name


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def boolish(series: pd.Series) -> pd.Series:
    """Convert common source flag encodings without treating unknown as true."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    values = series.astype("string").str.strip().str.lower()
    true_values = {"1", "true", "t", "yes", "y", "complete", "fully_observed", "observed"}
    return values.isin(true_values)


def normalise_status(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower()


def _first_existing(columns: Sequence[str], frame: pd.DataFrame) -> str | None:
    for name in columns:
        if name in frame.columns:
            return name
    return None


def covid_completeness(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    """Return value-complete mask, reason and the source flag used.

    The source panel may expose one of several documented coverage field names;
    this resolver accepts those variants while preserving a strict non-null
    fallback. A non-null partial/out-of-source status is never
    silently accepted.  Pre-COVID structural zeros are complete by design.
    """

    if COVID_RAW not in panel.columns:
        raise KeyError(f"panel_with_covid_v11.csv missing required field {COVID_RAW}")
    value = numeric(panel[COVID_RAW])
    value_ok = value.notna() & np.isfinite(value)
    status_col = _first_existing(
        (
            "covid_exposure_coverage_status_v11",
            "covid_exposure_status_v11",
            "covid_source_date_coverage_status",
            "covid_source_date_status",
            "covid_stringency_source_date_status",
            "covid_exposure_status",
        ),
        panel,
    )
    flag_col = _first_existing(
        (
            "covid_exposure_complete_v11",
            "covid_source_date_exposure_complete",
            "covid_source_date_complete",
            "covid_exposure_complete",
        ),
        panel,
    )
    n_col = _first_existing(
        (
            "covid_source_date_valid_n",
            "covid_exposure_valid_n",
            "covid_source_date_n_valid",
            "covid_exposure_n_valid",
        ),
        panel,
    )
    expected_col = _first_existing(
        (
            "covid_source_date_expected_n",
            "covid_exposure_expected_n",
            "covid_source_date_n_expected",
        ),
        panel,
    )

    complete = value_ok.copy()
    reason = pd.Series("complete_by_nonmissing_value", index=panel.index, dtype="string")
    if flag_col:
        flag = boolish(panel[flag_col])
        complete &= flag
        reason = np.where(flag, reason, f"{flag_col}_false_or_missing")
    if n_col and expected_col:
        valid_n = numeric(panel[n_col])
        expected_n = numeric(panel[expected_col])
        counts_ok = valid_n.eq(expected_n) & expected_n.gt(0)
        complete &= counts_ok
        reason = np.where(counts_ok, reason, f"{n_col}_not_equal_{expected_col}")
    if status_col:
        status = normalise_status(panel[status_col])
        partial_tokens = (
            "partial",
            "missing",
            "out_of_source",
            "out-of-source",
            "not_complete",
            "incomplete",
            "no_valid",
            "unavailable",
        )
        status_bad = status.str.contains("|".join(partial_tokens), regex=True, na=False)
        status_good = (~status_bad) & status.ne("")
        # An explicitly complete/observed status is sufficient, while an
        # empty status falls back to the value/flag checks above.
        complete &= ~status_bad
        reason = np.where(status_bad, f"{status_col}_partial_or_missing", reason)
        reason = np.where(status_good & ~status_bad, status_col + "_accepted", reason)
    # The source may carry a finite zero with an explicit structural-zero
    # label.  This remains complete and is not converted to missing.
    if status_col:
        status = normalise_status(panel[status_col])
        structural = status.str.contains("structural_zero|pre_covid", regex=True, na=False)
        complete |= structural & value_ok
        reason = np.where(structural & value_ok, "pre_covid_structural_zero", reason)
    complete = pd.Series(complete, index=panel.index).astype(bool)
    reason = pd.Series(reason, index=panel.index).astype("string")
    source = ",".join(x for x in [flag_col, n_col, expected_col, status_col] if x) or "nonmissing_value_fallback"
    return complete, reason, source


def load_sample() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the redistributable frozen S_COVID model input.

    The source build performed the charging/policy joins and coverage checks. This
    public entry point receives only the resulting eligible rows and keeps the
    row-level flags needed to audit the sample.  It never reads or reconstructs
    the restricted policy corpus.
    """

    if not PUBLIC_INPUT.exists():
        raise FileNotFoundError(f"Missing public model input: {PUBLIC_INPUT}")
    frame = read_csv(PUBLIC_INPUT).copy()
    required = {
        "analysis_row_id",
        "city_key",
        "city",
        "province",
        "year",
        "log_N_v10",
        "log_A_v10",
        "log_I_v10",
        *POLICY,
        *BASE_CONTROLS,
        COVID_RAW,
        "covid_exposure_complete_v11",
        "covid_model_eligible_v11",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"policy_model_input.csv missing required columns: {missing}")
    frame["year"] = numeric(frame["year"]).astype(int)
    frame["city_key"] = frame["city_key"].astype(str)
    frame["province"] = frame["province"].astype(str)
    numeric_cols = [*POLICY, *BASE_CONTROLS, "log_N_v10", "log_A_v10", "log_I_v10", COVID_RAW]
    for col in numeric_cols:
        frame[col] = numeric(frame[col])
    frame["covid_exposure_complete_v11"] = boolish(frame["covid_exposure_complete_v11"])
    frame["covid_candidate_year_le_2022_v11"] = frame["year"].le(2022)
    frame["covid_candidate_S427_v11"] = True
    frame["covid_candidate_v11"] = frame["covid_candidate_year_le_2022_v11"]
    frame["covid_model_eligible_v11"] = boolish(frame["covid_model_eligible_v11"])
    frame["covid_model_eligible_v11"] &= frame["covid_candidate_v11"] & frame["covid_exposure_complete_v11"]
    frame["covid_model_exclusion_reason_v11"] = ""
    frame.loc[~frame["covid_candidate_v11"], "covid_model_exclusion_reason_v11"] = "year_after_2022"
    frame.loc[frame["covid_candidate_v11"] & ~frame["covid_exposure_complete_v11"], "covid_model_exclusion_reason_v11"] = "incomplete_public_input_flag"
    complete_controls = frame[[*POLICY, *BASE_CONTROLS, "log_N_v10", "log_A_v10", "log_I_v10", COVID_RAW]].notna().all(axis=1)
    frame["covid_model_eligible_v11"] &= complete_controls
    frame.loc[~complete_controls, "covid_model_exclusion_reason_v11"] = "missing_model_X_or_Y"
    frame.sort_values(["city_key", "year", "analysis_row_id"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    eligible = frame.loc[frame["covid_model_eligible_v11"]].copy()
    eligible[COVID_SCALED] = eligible[COVID_RAW] / 10.0
    eligible["sample_id_v11"] = "S_COVID"
    if eligible.empty:
        raise ValueError("No eligible rows in policy_model_input.csv")
    support = {
        "candidate_rows_S427_year_le_2022": int(frame["covid_candidate_v11"].sum()),
        "candidate_cities": int(frame.loc[frame["covid_candidate_v11"], "city_key"].nunique()),
        "eligible_rows": int(len(eligible)),
        "eligible_cities": int(eligible["city_key"].nunique()),
        "eligible_provinces": int(eligible["province"].nunique()),
        "at_least_two_years_cities": int(eligible["city_key"].value_counts().ge(2).sum()),
        "R_switch_cities": int(_switch_cities(eligible, POLICY[0])),
        "S_switch_cities": int(_switch_cities(eligible, POLICY[1])),
        "excluded_candidate_rows": int(frame["covid_candidate_v11"].sum() - len(eligible)),
        "completeness_source": "eligible_flag_in_public_input",
        "sample_changed_by_source_gap": False,
    }
    return eligible, {"audit_frame": frame, "support": support, "completeness_source": support["completeness_source"]}


def _switch_cities(frame: pd.DataFrame, term: str) -> int:
    g = frame.groupby("city_key")[term].agg(["min", "max"])
    return int(((g["min"] < 1) & (g["max"] >= 1)).sum())


def qr_rank(matrix: np.ndarray) -> tuple[int, int, float, np.ndarray]:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n, p = arr.shape
    if p == 0:
        return 0, 0, 0.0, np.empty((0, 0))
    _, r = np.linalg.qr(arr, mode="reduced")
    diag = np.abs(np.diag(r)) if r.size else np.array([])
    scale = float(np.max(diag)) if len(diag) else 0.0
    tol = float(max(n, p) * np.finfo(float).eps * max(scale, 1.0) * 1_000.0)
    rank = int(np.sum(diag > tol))
    return rank, p, tol, r


def make_design(frame: pd.DataFrame, spec_id: str) -> tuple[np.ndarray, list[str], dict[str, str]]:
    if spec_id not in SPECS:
        raise ValueError(f"Unknown policy specification {spec_id}")
    terms = [*POLICY, *BASE_CONTROLS]
    if spec_id == "C1":
        terms.append(COVID_SCALED)
    parts: list[np.ndarray] = [np.ones((len(frame), 1), dtype=float)]
    names: list[str] = ["Intercept"]
    term_types: dict[str, str] = {"Intercept": "intercept"}
    for term in terms:
        vals = numeric(frame[term]).to_numpy(dtype=float)
        parts.append(vals.reshape(-1, 1))
        names.append(term)
        term_types[term] = "policy" if term in POLICY else ("covid" if term == COVID_SCALED else "control")
    cities = sorted(frame["city_key"].astype(str).unique().tolist())
    years = sorted(numeric(frame["year"]).astype(int).unique().tolist())
    for city in cities[1:]:
        names.append(f"city_fe:{city}")
        term_types[names[-1]] = "city_fe"
        parts.append(frame["city_key"].astype(str).eq(city).astype(float).to_numpy().reshape(-1, 1))
    for year in years[1:]:
        names.append(f"year_fe:{year}")
        term_types[names[-1]] = "year_fe"
        parts.append(numeric(frame["year"]).eq(year).astype(float).to_numpy().reshape(-1, 1))
    return np.column_stack(parts), names, term_types


def residual_rank(X: np.ndarray, names: Sequence[str], term: str) -> dict[str, Any]:
    """Rank and norm after projecting a focal term off all nuisance columns."""

    idx = names.index(term)
    nuisance = [i for i in range(len(names)) if i != idx]
    z = X[:, nuisance]
    p = X[:, [idx]]
    zrank, zdim, ztol, _ = qr_rank(z)
    if zrank == zdim:
        qz, _ = np.linalg.qr(z, mode="reduced")
        resid = p - qz @ (qz.T @ p)
    else:
        resid = p - z @ np.linalg.lstsq(z, p, rcond=None)[0]
    rank, dim, tol, _ = qr_rank(resid)
    return {
        "nuisance_rank": int(zrank),
        "nuisance_dim": int(zdim),
        "nuisance_tol": float(ztol),
        "residual_rank": int(rank),
        "residual_dim": int(dim),
        "residual_tol": float(tol),
        "residual_norm": float(np.linalg.norm(resid)),
        "identified": bool(rank == dim == 1),
    }


def policy_residual_ranks(X: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, term in (("R", POLICY[0]), ("S", POLICY[1])):
        rr = residual_rank(X, names, term)
        out[f"{label}_residual_rank"] = rr["residual_rank"]
        out[f"{label}_residual_dim"] = rr["residual_dim"]
        out[f"{label}_residual_norm"] = rr["residual_norm"]
        out[f"{label}_identified"] = rr["identified"]
    return out


def model_sample(frame: pd.DataFrame, spec_id: str) -> pd.DataFrame:
    sub = frame.copy()
    if spec_id == "C1":
        sub[COVID_SCALED] = numeric(sub[COVID_RAW]) / 10.0
    # C0 and C1 must have the same rows and same base X.  Calling this check
    # separately makes a future accidental spec-specific filter fail loudly.
    needed = ["log_N_v10", "log_A_v10", "log_I_v10", *POLICY, *BASE_CONTROLS]
    if spec_id == "C1":
        needed.append(COVID_SCALED)
    complete = sub[needed].notna().all(axis=1)
    sub = sub.loc[complete].copy()
    sub.sort_values(["city_key", "year", "analysis_row_id"], inplace=True)
    sub.reset_index(drop=True, inplace=True)
    return sub


def spec_table(frame: pd.DataFrame, samples: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    for spec_id in SPECS:
        sub = samples[spec_id]
        X, names, term_types = make_design(sub, spec_id)
        rank, dim, tol, _ = qr_rank(X)
        cov_rr = residual_rank(X, names, COVID_SCALED) if spec_id == "C1" else {
            "residual_rank": np.nan,
            "residual_dim": np.nan,
            "residual_norm": np.nan,
            "identified": np.nan,
        }
        policy_rr = policy_residual_ranks(X, names)
        province = _province_cluster(sub)
        counts = sub["city_key"].value_counts()
        diag: dict[str, Any] = {
            "spec_id": spec_id,
            "n": int(len(sub)),
            "Gp": int(province.nunique()),
            "G_city": int(sub["city_key"].nunique()),
            "years": int(sub["year"].nunique()),
            "rank": int(rank),
            "dim": int(dim),
            "rank_tol": float(tol),
            "rank_status": "full_rank" if rank == dim else "rank_deficient",
            "design_columns": names,
            "term_types": term_types,
            "covid_residual_rank": cov_rr["residual_rank"],
            "covid_residual_dim": cov_rr["residual_dim"],
            "covid_residual_norm": cov_rr["residual_norm"],
            "covid_identified": bool(cov_rr["identified"]) if spec_id == "C1" else np.nan,
            "covid_identification_status": "identified" if spec_id == "C1" and bool(cov_rr["identified"]) else "not_applicable",
            **policy_rr,
            "singletons": int(counts.eq(1).sum()),
            "two_or_more_years": int(counts.ge(2).sum()),
        }
        diagnostics[spec_id] = diag
        formula_terms = [*POLICY, *BASE_CONTROLS] + ([f"{COVID_RAW}/10"] if spec_id == "C1" else [])
        formula = "log(y_outcome) ~ " + " + ".join(formula_terms) + " + cityFE + yearFE"
        for outcome in OUTCOMES:
            rows.append(
                {
                    "spec_id": spec_id,
                    "model_id": f"COVID_{spec_id}_{outcome}",
                    "outcome": outcome,
                    "sample_id": "S_COVID",
                    "sample_rule": "V10 S427 and year<=2022 and compatible-date COVID exposure complete",
                    "formula": formula,
                    "outcome_definition": {"N": "log(E/W)", "A": "log(Q/W)", "I": "log(Q/E)"}[outcome],
                    "policy_terms": "+".join(POLICY),
                    "base_controls": "+".join(BASE_CONTROLS),
                    "covid_term": f"{COVID_RAW}/10" if spec_id == "C1" else "",
                    "fixed_effects": "city_key+year",
                    "cluster_level": "province",
                    "n": int(diag["n"]),
                    "G_city": int(diag["G_city"]),
                    "Gp": int(diag["Gp"]),
                    "rank": int(diag["rank"]),
                    "dim": int(diag["dim"]),
                    "df": int(diag["Gp"] - 1),
                    "rank_status": diag["rank_status"],
                    "covid_residual_rank": diag["covid_residual_rank"],
                    "covid_residual_dim": diag["covid_residual_dim"],
                    "covid_residual_norm": diag["covid_residual_norm"],
                    "covid_identified": diag["covid_identified"] if spec_id == "C1" else np.nan,
                    "covid_identification_status": diag["covid_identification_status"],
                    "residual_R_rank": diag["R_residual_rank"],
                    "residual_S_rank": diag["S_residual_rank"],
                    "singletons": int(diag["singletons"]),
                    "two_or_more_years": int(diag["two_or_more_years"]),
                    "fit_status": "prefit_ready" if (rank == dim and (spec_id == "C0" or diag["covid_identified"])) else "not_estimable",
                    "se_method": "CR1_province_finite_correction",
                    "reference_distribution": "t(Gp-1)",
                    "frozen_before_fit": True,
                    "math_version": MATH_VERSION,
                }
            )
    return pd.DataFrame(rows), diagnostics


def _province_cluster(frame: pd.DataFrame) -> pd.Series:
    """Stable province labels for shared-background CR1 clustering.

    The ``province`` field is the canonical city-crosswalk label and has the
    25-province support expected by S_COVID. Fallbacks are retained for a
    panel that exposes a differently named province key.
    """

    for col in ("province", "province_key_v9", "province_key", "province_name", "province_code"):
        if col in frame.columns:
            vals = frame[col].astype("string").fillna("").str.strip()
            if vals.ne("").all():
                return vals.astype(str)
    raise KeyError("No nonmissing province field available for province clustering")


def outcome_arrays(frame: pd.DataFrame) -> np.ndarray:
    return frame[["log_N_v10", "log_A_v10", "log_I_v10"]].to_numpy(dtype=float)


def fit_qr(X: np.ndarray, Y: np.ndarray, groups: pd.Series) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n, p = X.shape
    rank, dim, tol, _ = qr_rank(X)
    if rank != dim:
        raise np.linalg.LinAlgError(f"rank deficient full design rank={rank}, dim={dim}")
    q, r = np.linalg.qr(X, mode="reduced")
    beta = np.linalg.solve(r, q.T @ Y)
    residual = Y - X @ beta
    groups_arr = groups.astype(str).to_numpy()
    unique_groups = sorted(pd.unique(groups_arr).tolist())
    Gp = len(unique_groups)
    if Gp <= 1 or n <= p:
        raise np.linalg.LinAlgError(f"invalid province CR1 dimensions n={n}, K={p}, Gp={Gp}")
    bread = np.linalg.inv(r) @ np.linalg.inv(r).T
    correction = (Gp / (Gp - 1)) * ((n - 1) / (n - p))
    covariance_slices: list[np.ndarray] = []
    for outcome_idx in range(Y.shape[1]):
        meat = np.zeros((p, p), dtype=float)
        for group in unique_groups:
            ix = groups_arr == group
            score = X[ix].T @ residual[ix, outcome_idx]
            meat += np.outer(score, score)
        cov = correction * (bread @ meat @ bread)
        covariance_slices.append((cov + cov.T) / 2.0)
    covariance = np.stack(covariance_slices, axis=2)
    return beta, covariance, {
        "n": int(n),
        "K": int(p),
        "Gp": int(Gp),
        "rank": int(rank),
        "dim": int(dim),
        "df": int(Gp - 1),
        "rank_tol": float(tol),
        "cr1_correction": float(correction),
        "residual": residual,
        "groups": unique_groups,
    }


def synthetic_covariance_check() -> dict[str, Any]:
    """Exercise covariance dimensions before reading/fitting the real sample."""

    rng = np.random.default_rng(20261103)
    n, p, g = 20, 5, 5
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    groups = pd.Series(np.repeat([f"p{i}" for i in range(g)], n // g))
    Y = X @ rng.normal(size=(p, 3)) + rng.normal(scale=0.1, size=(n, 3))
    beta, cov, meta = fit_qr(X, Y, groups)
    assert beta.shape == (p, 3)
    assert cov.shape == (p, p, 3)
    assert np.isfinite(beta).all() and np.isfinite(cov).all()
    assert all(np.allclose(cov[:, :, i], cov[:, :, i].T, atol=1e-12) for i in range(3))
    return {"passed": True, "n": n, "K": p, "Gp": g, "covariance_shape": list(cov.shape), "cr1_correction": meta["cr1_correction"]}


def term_rows(
    spec_id: str,
    outcome: str,
    names: Sequence[str],
    term_types: Mapping[str, str],
    beta: np.ndarray,
    covariance: np.ndarray,
    meta: Mapping[str, Any],
    status: str = "estimated",
    failure_reason: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    df = int(meta.get("df", 0))
    crit = float(stats.t.ppf(0.975, df)) if df > 0 else np.nan
    for j, term in enumerate(names):
        est = float(beta[j]) if status == "estimated" else np.nan
        var = float(covariance[j, j]) if status == "estimated" and np.isfinite(covariance[j, j]) else np.nan
        se = float(np.sqrt(max(var, 0.0))) if np.isfinite(var) else np.nan
        tstat = est / se if np.isfinite(est) and np.isfinite(se) and se > 0 else np.nan
        pvalue = float(2 * stats.t.sf(abs(tstat), df)) if np.isfinite(tstat) and df > 0 else np.nan
        rows.append(
            {
                "spec_id": spec_id,
                "model_id": f"COVID_{spec_id}_{outcome}",
                "outcome": outcome,
                "term": term,
                "term_type": term_types.get(term, "unknown"),
                "estimate": est,
                "se": se,
                "std_error": se,
                "t_stat": tstat,
                "p_value": pvalue,
                "ci_low": est - crit * se if np.isfinite(est) and np.isfinite(se) else np.nan,
                "ci_high": est + crit * se if np.isfinite(est) and np.isfinite(se) else np.nan,
                "n": int(meta.get("n", 0)),
                "K": int(meta.get("K", 0)),
                "Gp": int(meta.get("Gp", 0)),
                "rank": int(meta.get("rank", 0)),
                "dim": int(meta.get("dim", 0)),
                "df": df,
                "se_method": "CR1_province_finite_correction",
                "cluster_level": "province",
                "cr1_correction": float(meta.get("cr1_correction", np.nan)),
                "status": status,
                "failure_reason": failure_reason,
            }
        )
    return rows


def fit_all(
    specs: pd.DataFrame,
    diagnostics: Mapping[str, Mapping[str, Any]],
    samples: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    result_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    execution: list[dict[str, Any]] = []
    for spec_id in SPECS:
        sub = samples[spec_id]
        X, names, term_types = make_design(sub, spec_id)
        groups = _province_cluster(sub)
        diag = diagnostics[spec_id]
        try:
            beta, covariance, meta = fit_qr(X, outcome_arrays(sub), groups)
            status = "estimated"
            reason = ""
        except Exception as exc:
            beta = np.full((len(names), 3), np.nan)
            covariance = np.full((len(names), len(names), 3), np.nan)
            meta = {
                "n": len(sub),
                "K": X.shape[1],
                "Gp": groups.nunique(),
                "rank": diag["rank"],
                "dim": diag["dim"],
                "df": groups.nunique() - 1,
                "cr1_correction": np.nan,
            }
            status = "not_estimable"
            reason = f"{type(exc).__name__}: {exc}"
        signature = signature_for(X, outcome_arrays(sub), groups)
        for out_idx, outcome in enumerate(OUTCOMES):
            cov = covariance[:, :, out_idx] if status == "estimated" else np.full((len(names), len(names)), np.nan)
            b = beta[:, out_idx] if status == "estimated" else np.full(len(names), np.nan)
            # The QR call solves all three compatible outcomes together, but
            # the append-only audit records each logical OLS model separately
            # so the six-model execution count is directly inspectable.
            execution.append(
                {
                    "timestamp_utc": utc_now(),
                    "model_id": f"COVID_{spec_id}_{outcome}",
                    "spec_id": spec_id,
                    "outcome": outcome,
                    "status": status,
                    "signature": signature,
                    "n": int(meta["n"]),
                    "K": int(meta["K"]),
                    "Gp": int(meta["Gp"]),
                    "rank": int(meta["rank"]),
                    "math_version": MATH_VERSION,
                    "solve_count": 1 if status == "estimated" else 0,
                    "joint_qr_batch": f"{spec_id}_N_A_I",
                    "failure_reason": reason,
                }
            )
            result_rows.extend(term_rows(spec_id, outcome, names, term_types, b, cov, meta, status, reason))
            for i, term_i in enumerate(names):
                for j, term_j in enumerate(names):
                    covariance_rows.append(
                        {
                            "spec_id": spec_id,
                            "model_id": f"COVID_{spec_id}_{outcome}",
                            "outcome": outcome,
                            "term_i": term_i,
                            "term_j": term_j,
                            "covariance": float(cov[i, j]) if np.isfinite(cov[i, j]) else np.nan,
                            "n": int(meta["n"]),
                            "K": int(meta["K"]),
                            "Gp": int(meta["Gp"]),
                            "rank": int(meta["rank"]),
                            "df": int(meta["df"]),
                            "cluster_level": "province",
                            "status": status,
                        }
                    )
    return pd.DataFrame(result_rows), pd.DataFrame(covariance_rows), execution


def signature_for(X: np.ndarray, Y: np.ndarray, groups: pd.Series) -> str:
    """Hash only actual arrays and the estimator math version."""

    h = hashlib.sha256()
    h.update(MATH_VERSION.encode("utf-8"))
    for arr in (np.asarray(X, dtype="<f8"), np.asarray(Y, dtype="<f8")):
        h.update(str(arr.shape).encode("ascii"))
        h.update(np.ascontiguousarray(arr).tobytes())
    group_values = np.asarray(groups.astype(str).tolist(), dtype="U")
    h.update(str(group_values.shape).encode("ascii"))
    h.update(group_values.tobytes())
    return h.hexdigest()


def identity_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec_id in SPECS:
        one = results.loc[results["spec_id"].eq(spec_id)]
        for term in sorted(one["term"].astype(str).unique()):
            vals: dict[str, float] = {}
            for outcome in OUTCOMES:
                hit = one.loc[(one["outcome"].eq(outcome)) & (one["term"].eq(term))]
                vals[outcome] = float(hit.iloc[0]["estimate"]) if not hit.empty else np.nan
            diff = vals["A"] - vals["N"] if np.isfinite(vals["A"]) and np.isfinite(vals["N"]) else np.nan
            err = vals["I"] - diff if np.isfinite(vals["I"]) and np.isfinite(diff) else np.nan
            rows.append({"spec_id": spec_id, "term": term, "beta_N": vals["N"], "beta_A": vals["A"], "beta_I": vals["I"], "beta_A_minus_beta_N": diff, "identity_error": err, "identity_abs_error": abs(err) if np.isfinite(err) else np.nan, "identity_pass": bool(np.isfinite(err) and abs(err) <= 1e-9)})
    return pd.DataFrame(rows)


def model_comparison(results: pd.DataFrame) -> pd.DataFrame:
    keep = results[["spec_id", "outcome", "term", "estimate", "se", "ci_low", "ci_high", "p_value", "n", "K", "Gp", "rank", "status"]].copy()
    c0 = keep.loc[keep.spec_id.eq("C0")].drop(columns="spec_id").rename(columns={c: f"{c}_C0" for c in keep.columns if c not in {"spec_id", "outcome", "term"}})
    c1 = keep.loc[keep.spec_id.eq("C1")].drop(columns="spec_id").rename(columns={c: f"{c}_C1" for c in keep.columns if c not in {"spec_id", "outcome", "term"}})
    out = c0.merge(c1, on=["outcome", "term"], how="outer", validate="one_to_one")
    out["estimate_C1_minus_C0"] = out["estimate_C1"] - out["estimate_C0"]
    out["ci_low_change"] = out["ci_low_C1"] - out["ci_low_C0"]
    out["ci_high_change"] = out["ci_high_C1"] - out["ci_high_C0"]
    out["comparison_note"] = np.where(out["term"].eq(COVID_SCALED), "C1-only COVID stringency coefficient; C0 omitted by design", "same S_COVID and same base X; C1 adds COVID/10")
    return out.sort_values(["outcome", "term"]).reset_index(drop=True)


def support_tables(audit: pd.DataFrame, eligible: pd.DataFrame, support: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, value in support.items():
        rows.append({"support_group": "S_COVID", "metric": key, "value": value, "n": len(eligible), "cities": eligible.city_key.nunique(), "provinces": _province_cluster(eligible).nunique(), "notes": "actual sample; candidate/source-gap differences retained"})
    reason_counts = audit.loc[audit["covid_candidate_v11"] & ~audit["covid_model_eligible_v11"], "covid_model_exclusion_reason_v11"].value_counts(dropna=False)
    for reason, count in reason_counts.items():
        rows.append({"support_group": "exclusion", "metric": "excluded_rows_by_reason", "value": int(count), "n": len(eligible), "cities": np.nan, "provinces": np.nan, "notes": str(reason)})
    for term in POLICY:
        rows.append({"support_group": "transitions", "metric": f"{term}_switch_cities", "value": _switch_cities(eligible, term), "n": len(eligible), "cities": eligible.city_key.nunique(), "provinces": _province_cluster(eligible).nunique(), "notes": "city has both 0 and 1 in S_COVID"})
    return pd.DataFrame(rows)


def load_history() -> list[dict[str, Any]]:
    path = OUTPUT / "model_execution_history_v11.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return payload["records"]
        if isinstance(payload, list):
            return payload
    except Exception:
        return []
    return []


def cached_payload(signatures: Mapping[str, str]) -> bool:
    metadata_path = OUTPUT / "model_metadata_v11.json"
    required = [OUTPUT / x for x in ["covid_model_results_v11.csv", "covid_model_covariance_v11.csv", "covid_model_comparison_v11.csv", "covid_model_support_v11.csv"]]
    if not metadata_path.exists() or not all(p.exists() for p in required):
        return False
    try:
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return old.get("fit_signatures") == dict(signatures) and old.get("math_version") == MATH_VERSION


def main(*, force_fit: bool = False) -> dict[str, Any]:
    synthetic = synthetic_covariance_check()
    frame, info = load_sample()
    # C0/C1 must be exactly the same row order, outcomes and base covariates.
    samples = {spec_id: model_sample(frame, spec_id) for spec_id in SPECS}
    if not samples["C0"]["analysis_row_id"].equals(samples["C1"]["analysis_row_id"]):
        raise AssertionError("C0 and C1 do not share identical S_COVID row order")
    if not samples["C0"]["city_key"].equals(samples["C1"]["city_key"]):
        raise AssertionError("C0 and C1 do not share identical city ordering")
    for spec_id in SPECS:
        samples[spec_id][COVID_SCALED] = numeric(samples[spec_id][COVID_RAW]) / 10.0
    specs, diagnostics = spec_table(frame, samples)
    # Freeze model_specs before any estimator call.
    specs_path = OUTPUT / "model_specs_v11.csv"
    specs.to_csv(specs_path, index=False, encoding="utf-8-sig")
    signatures = {
        spec_id: signature_for(*((lambda sub: (make_design(sub, spec_id)[0], outcome_arrays(sub), _province_cluster(sub)))(samples[spec_id])))
        for spec_id in SPECS
    }
    # The lambda above is intentionally only an array assembly; it performs no
    # fit.  Keep deterministic design diagnostics in metadata without adding
    # prose or script hashes to the fit signature.
    fit_cache = (not force_fit) and cached_payload(signatures)
    history = load_history()
    fit_mode = "cache_reuse"
    if fit_cache:
        results = read_csv(OUTPUT / "covid_model_results_v11.csv")
        covariance = read_csv(OUTPUT / "covid_model_covariance_v11.csv")
        execution: list[dict[str, Any]] = []
    else:
        results, covariance, execution = fit_all(specs, diagnostics, samples)
        results.to_csv(OUTPUT / "covid_model_results_v11.csv", index=False, encoding="utf-8-sig")
        covariance.to_csv(OUTPUT / "covid_model_covariance_v11.csv", index=False, encoding="utf-8-sig")
        history.extend(execution)
        fit_mode = "fresh_fit"
    # Prefit rows are frozen before estimation; after a fit/cache read, only
    # this deterministic status annotation is refreshed.  It never changes
    # the sample, X/Y arrays or fit signature.
    status_by_model = results.groupby("model_id")["status"].first().to_dict()
    specs["fit_status"] = specs["model_id"].map(status_by_model).fillna(specs["fit_status"])
    specs.to_csv(specs_path, index=False, encoding="utf-8-sig")
    # All rows below are deterministic post-fit/report artifacts.  They may be
    # regenerated on a cache hit without touching execution history.
    comparison = model_comparison(results)
    comparison.to_csv(OUTPUT / "covid_model_comparison_v11.csv", index=False, encoding="utf-8-sig")
    support = support_tables(info["audit_frame"], frame, info["support"])
    support.to_csv(OUTPUT / "covid_model_support_v11.csv", index=False, encoding="utf-8-sig")
    identity = identity_table(results)
    identity.to_csv(OUTPUT / "covid_model_identity_v11.csv", index=False, encoding="utf-8-sig")
    # The model sample is the actual selected S_COVID frame, not a synthetic
    # reconstruction.  The complete candidate audit is included in the same
    # file as a separate CSV so exclusions are recoverable without refitting.
    frame.to_csv(OUTPUT / "covid_model_sample_v11.csv", index=False, encoding="utf-8-sig")
    info["audit_frame"].to_csv(OUTPUT / "covid_model_sample_audit_v11.csv", index=False, encoding="utf-8-sig")
    history_payload = {"math_version": MATH_VERSION, "records": history}
    write_json(OUTPUT / "model_execution_history_v11.json", history_payload)
    # A plainly named alias is useful to downstream validators; both files
    # contain the same append-only records.
    write_json(OUTPUT / "execution_history_v11.json", history_payload)
    metadata = {
        "generated_utc": utc_now(),
        "script": rel(HERE),
        "python": platform.python_version(),
        "fit_mode": fit_mode,
        "math_version": MATH_VERSION,
        "fit_signatures": signatures,
        "execution_counts": {
            "main_model_fits_this_run": 0 if fit_cache else int(sum(r.get("solve_count", 0) for r in execution)),
            "main_model_fits_planned": 6,
            "main_model_fits_total_recorded": int(sum(r.get("solve_count", 0) for r in history)),
            "covariance_solves_this_run": 0 if fit_cache else 2,
            "synthetic_validation_solves_this_run": 1,
        },
        "sample_support": info["support"],
        "diagnostics": diagnostics,
        "inference": {"se_method": "CR1_province_finite_correction", "cr1_formula": "(Gp/(Gp-1))*((n-1)/(n-K))", "K_definition": "full QR design rank/dimension", "reference_distribution": "t(Gp-1)", "cluster_level": "province"},
        "guardrails": {"same_sample_C0_C1": True, "same_X_three_outcomes": True, "no_GDP_or_area": True, "no_JK_or_bootstrap": True, "full_rank_checked": True, "covid_residual_identification_checked": True, "synthetic_covariance_shape_checked": synthetic["passed"]},
        "inputs": {"public_policy_model_input": rel(PUBLIC_INPUT), "public_policy_model_input_sha256": sha256(PUBLIC_INPUT)},
        "outputs": {p.name: {"path": rel(p), "size_bytes": int(p.stat().st_size)} for p in OUTPUT.glob("covid_model_*.csv")},
    }
    write_json(OUTPUT / "model_metadata_v11.json", metadata)
    print(json.dumps({"fit_mode": fit_mode, "S_COVID_rows": len(frame), "S_COVID_cities": frame.city_key.nunique(), "S_COVID_provinces": _province_cluster(frame).nunique(), "models": 6, "actual_solves_this_run": metadata["execution_counts"]["main_model_fits_this_run"]}, ensure_ascii=False))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the six policy-association models")
    parser.add_argument("--force-fit", action="store_true", help="Ignore a valid fit cache and refit")
    args = parser.parse_args()
    main(force_fit=args.force_fit)

