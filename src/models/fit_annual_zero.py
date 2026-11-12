"""Estimate the annual fixed-elasticity PPML models with and without zeros.

The public entry point reads ``data/processed/annual_service_metrics_v9.csv``
and ``data/processed/weather_city_year_v4.csv``.  It writes the frozen sample
audit, specifications, coefficient tables, full covariance matrices, numeric
fit log, and status record under ``data/results/recomputed_annual_zero/``.
It fits exactly the two prespecified Poisson QMLE specifications (unless an
explicitly logged numerical equivalent repair is required):

    E[Q_it | X, W] = W_it * exp(city_FE + year_FE + beta ln(N_it)
                                + gamma' weather_it)

``P0_positive`` uses the positive annual support (445 rows and 104 cities
after the city-FE singleton rule).  ``P0_with_zeros`` starts from all rows
with finite non-negative Q, positive E and W, finite positive N, and complete
weather controls.  It removes all-zero candidate cities first, and then
singleton cities.  Those exclusions remain in the sample audit and are never
silently discarded from the descriptive support.

Inference follows the established implementation: the Poisson score and
Hessian are used for a city-cluster CR1 sandwich, with a t(G-1) interval.
W is an offset exposure (ln W), not a frequency weight.  No bootstrap or
additional PPML specification is run here.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import scipy
import statsmodels
import statsmodels.api as sm


HERE = Path(__file__).resolve()
REPOSITORY = HERE.parents[2]
OUTPUT = REPOSITORY / "data" / "results" / "recomputed_annual_zero"
ANNUAL_INPUT = REPOSITORY / "data" / "processed" / "annual_service_metrics_v9.csv"
WEATHER_INPUT = REPOSITORY / "data" / "processed" / "weather_city_year_v4.csv"

MODEL_IDS = ("P0_positive", "P0_with_zeros")
WEATHER_CONTROLS = (
    "weather_hdd18_degree_days",
    "weather_cdd24_degree_days",
    "weather_precipitation_sum_mm",
    "weather_wind_speed_mean_ms",
    "weather_relative_humidity_mean_pct",
)
SCRIPT_VERSION = "v15_p0_two_spec_ppml_score_hessian_2026-11-06"
MODEL_FORMULA = (
    "E[Q_it|X,W]=W_it*exp(alpha_i + lambda_t + beta*ln(N_it) "
    "+ gamma'weather_it)"
)
SAMPLE_RULE_ZERO = (
    "Q non-missing and >=0; W>0; E>0; finite positive N=E/W; "
    "five weather controls complete"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY.resolve()).as_posix()
    except ValueError:
        return path.name


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(dict(payload)), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _reason_list(row: pd.Series) -> str:
    """List every preflight failure so no non-candidate row is silent."""

    reasons: list[str] = []
    if not bool(row["q_nonmissing"]):
        reasons.append("Q_missing")
    elif not bool(row["q_nonnegative"]):
        reasons.append("Q_negative")
    if not bool(row["w_positive"]):
        reasons.append("W_nonpositive_or_missing")
    if not bool(row["e_positive"]):
        reasons.append("E_nonpositive_or_missing")
    if not bool(row["n_finite_positive"]):
        reasons.append("N_not_finite_positive")
    if not bool(row["weather_complete"]):
        reasons.append("weather_incomplete")
    return ";".join(reasons) if reasons else ""


def build_samples() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build and audit the full panel, positive model, and zero model samples."""

    if not ANNUAL_INPUT.exists() or not WEATHER_INPUT.exists():
        raise FileNotFoundError(f"Missing PPML inputs: {ANNUAL_INPUT}, {WEATHER_INPUT}")
    annual = read_csv(ANNUAL_INPUT)
    weather = read_csv(WEATHER_INPUT)
    required_annual = {"analysis_row_id", "city_key", "city", "province", "year", "Q", "E", "W"}
    missing = sorted(required_annual.difference(annual.columns))
    if missing:
        raise KeyError(f"annual_service_metrics_v9 missing columns: {missing}")
    missing_weather = sorted(set(WEATHER_CONTROLS).difference(weather.columns))
    if missing_weather:
        raise KeyError(f"weather_city_year_v4 missing fixed controls: {missing_weather}")
    if annual.duplicated(["city_key", "year"]).any():
        raise ValueError("annual_service_metrics_v9 is not unique on city_key-year")
    if weather.duplicated(["city_key", "year"]).any():
        raise ValueError("weather_city_year_v4 is not unique on city_key-year")

    keep_annual = [
        "analysis_row_id",
        "city_key",
        "city",
        "province",
        "year",
        "source_city",
        "Q",
        "E",
        "W",
        "q_status",
        "e_status",
        "w_status",
        "q_zero_semantics",
        "e_zero_semantics",
        "service_status",
    ]
    keep_annual = [c for c in keep_annual if c in annual.columns]
    panel = annual[keep_annual].copy()
    panel["year"] = numeric(panel["year"]).astype("Int64")
    for col in ("Q", "E", "W"):
        panel[col] = numeric(panel[col])

    weather_keep = weather[["city_key", "year", *WEATHER_CONTROLS]].copy()
    weather_keep["year"] = numeric(weather_keep["year"]).astype("Int64")
    for col in WEATHER_CONTROLS:
        weather_keep[col] = numeric(weather_keep[col])
    panel = panel.merge(weather_keep, on=["city_key", "year"], how="left", indicator=True, validate="one_to_one")
    panel["weather_key_match"] = panel["_merge"].eq("both")
    panel.drop(columns=["_merge"], inplace=True)

    q = panel["Q"].to_numpy(float)
    e = panel["E"].to_numpy(float)
    w = panel["W"].to_numpy(float)
    n = np.divide(e, w, out=np.full(len(panel), np.nan), where=np.isfinite(e) & np.isfinite(w) & (w != 0))
    panel["N"] = n
    with np.errstate(divide="ignore", invalid="ignore"):
        panel["lnN"] = np.where(np.isfinite(n) & (n > 0), np.log(n), np.nan)
        panel["lnW"] = np.where(np.isfinite(w) & (w > 0), np.log(w), np.nan)
    panel["q_nonmissing"] = np.isfinite(q)
    panel["q_nonnegative"] = panel["q_nonmissing"] & (q >= 0)
    panel["w_positive"] = np.isfinite(w) & (w > 0)
    panel["e_positive"] = np.isfinite(e) & (e > 0)
    panel["n_finite_positive"] = np.isfinite(n) & (n > 0)
    panel["weather_complete"] = panel[list(WEATHER_CONTROLS)].notna().all(axis=1)
    panel["candidate_preflight"] = (
        panel["q_nonnegative"]
        & panel["w_positive"]
        & panel["e_positive"]
        & panel["n_finite_positive"]
        & panel["weather_complete"]
    )
    panel["source_encoded_zero"] = panel["candidate_preflight"] & panel["Q"].eq(0)
    panel["candidate_q_state"] = np.select(
        [panel["source_encoded_zero"], panel["candidate_preflight"] & panel["Q"].gt(0)],
        ["observed_compatible_zero", "positive"],
        default="not_candidate",
    )
    panel["preflight_exclusion_reason"] = panel.apply(_reason_list, axis=1)

    candidate = panel.loc[panel["candidate_preflight"]].copy()
    candidate_cities = candidate["city_key"].astype(str).unique().tolist()
    candidate_city_q = candidate.groupby("city_key", sort=False)["Q"].sum(min_count=1)
    all_zero_cities = set(candidate_city_q.index[candidate_city_q.eq(0)].astype(str))
    panel["candidate_all_zero_city"] = panel["city_key"].astype(str).isin(all_zero_cities)
    panel["after_all_zero_removal"] = panel["candidate_preflight"] & ~panel["candidate_all_zero_city"]

    after_all_zero = candidate.loc[~candidate["city_key"].astype(str).isin(all_zero_cities)].copy()
    after_counts = after_all_zero.groupby("city_key", sort=False).size()
    singleton_zero_cities = set(after_counts.index[after_counts.eq(1)].astype(str))
    panel["p0_with_zeros_singleton_city"] = panel["city_key"].astype(str).isin(singleton_zero_cities)
    panel["p0_with_zeros_model_member"] = panel["after_all_zero_removal"] & ~panel["p0_with_zeros_singleton_city"]
    panel["p0_with_zeros_exclusion_reason"] = panel["preflight_exclusion_reason"]
    panel.loc[panel["candidate_preflight"] & panel["candidate_all_zero_city"], "p0_with_zeros_exclusion_reason"] = "all_zero_candidate_city_FE_separation"
    panel.loc[panel["after_all_zero_removal"] & panel["p0_with_zeros_singleton_city"], "p0_with_zeros_exclusion_reason"] = "singleton_city_no_within_city_information"
    panel.loc[panel["p0_with_zeros_model_member"], "p0_with_zeros_exclusion_reason"] = ""

    positive_candidate = panel.loc[panel["candidate_preflight"] & panel["Q"].gt(0)].copy()
    positive_counts = positive_candidate.groupby("city_key", sort=False).size()
    positive_singleton_cities = set(positive_counts.index[positive_counts.eq(1)].astype(str))
    panel["p0_positive_candidate"] = panel["candidate_preflight"] & panel["Q"].gt(0)
    panel["p0_positive_singleton_city"] = panel["city_key"].astype(str).isin(positive_singleton_cities)
    panel["p0_positive_model_member"] = panel["p0_positive_candidate"] & ~panel["p0_positive_singleton_city"]
    panel["p0_positive_exclusion_reason"] = panel["preflight_exclusion_reason"]
    panel.loc[panel["candidate_preflight"] & panel["Q"].eq(0), "p0_positive_exclusion_reason"] = "Q_zero_not_positive_specification"
    panel.loc[panel["p0_positive_candidate"] & panel["p0_positive_singleton_city"], "p0_positive_exclusion_reason"] = "singleton_positive_city_no_within_city_information"
    panel.loc[panel["p0_positive_model_member"], "p0_positive_exclusion_reason"] = ""

    panel["p0_with_zeros_sample_role"] = np.select(
        [panel["p0_with_zeros_model_member"], panel["candidate_preflight"]],
        ["P0_with_zeros_model", "P0_with_zeros_excluded_candidate"],
        default="P0_with_zeros_outside_candidate",
    )
    panel["p0_positive_sample_role"] = np.select(
        [panel["p0_positive_model_member"], panel["p0_positive_candidate"]],
        ["P0_positive_model", "P0_positive_excluded_candidate"],
        default="P0_positive_outside_candidate",
    )
    panel["sample_rule_version"] = "V15_D_P0_deterministic_candidate_then_allzero_then_singleton"
    panel.sort_values(["city_key", "year", "analysis_row_id"], kind="stable", inplace=True, ignore_index=True)

    model_frames = {
        "P0_positive": panel.loc[panel["p0_positive_model_member"]].copy(),
        "P0_with_zeros": panel.loc[panel["p0_with_zeros_model_member"]].copy(),
    }
    info: dict[str, Any] = {
        "panel_rows": int(len(panel)),
        "panel_cities": int(panel["city_key"].nunique()),
        "panel_years": sorted(panel["year"].dropna().astype(int).unique().tolist()),
        "candidate_rows": int(candidate.shape[0]),
        "candidate_cities": int(candidate["city_key"].nunique()),
        "candidate_positive_rows": int(candidate["Q"].gt(0).sum()),
        "candidate_zero_rows": int(candidate["Q"].eq(0).sum()),
        "candidate_missing_weather_rows": int(
            ((panel["q_nonnegative"] & panel["w_positive"] & panel["e_positive"] & panel["n_finite_positive"]) & ~panel["weather_complete"]).sum()
        ),
        "all_zero_city_count": int(len(all_zero_cities)),
        "all_zero_rows": int(candidate["city_key"].astype(str).isin(all_zero_cities).sum()),
        "after_all_zero_rows": int(after_all_zero.shape[0]),
        "after_all_zero_cities": int(after_all_zero["city_key"].nunique()),
        "singleton_zero_city_count": int(len(singleton_zero_cities)),
        "singleton_zero_rows": int(after_all_zero["city_key"].astype(str).isin(singleton_zero_cities).sum()),
        "positive_candidate_rows": int(positive_candidate.shape[0]),
        "positive_candidate_cities": int(positive_candidate["city_key"].nunique()),
        "positive_singleton_city_count": int(len(positive_singleton_cities)),
        "positive_singleton_rows": int(positive_candidate["city_key"].astype(str).isin(positive_singleton_cities).sum()),
        "controls": list(WEATHER_CONTROLS),
        "candidate_rule": SAMPLE_RULE_ZERO,
        "all_zero_rule": "remove cities whose candidate Q sum equals zero before singleton rule",
        "singleton_rule": "remove remaining cities with exactly one candidate row for city FE within-city information",
        "candidate_city_keys": candidate_cities,
        "all_zero_city_keys": sorted(all_zero_cities),
        "singleton_zero_city_keys": sorted(singleton_zero_cities),
        "positive_singleton_city_keys": sorted(positive_singleton_cities),
        "model_frames": model_frames,
    }
    return panel, candidate, info


def make_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, str], dict[str, Any]]:
    """Construct a full-rank city/year FE design without hard-coded degrees."""

    f = frame.sort_values(["city_key", "year", "analysis_row_id"], kind="stable").reset_index(drop=True)
    blocks: list[pd.DataFrame] = [pd.DataFrame({"const": np.ones(len(f)), "lnN": f["lnN"].to_numpy(float)})]
    term_types: dict[str, str] = {"const": "intercept", "lnN": "scale_elasticity"}
    controls = f[list(WEATHER_CONTROLS)].astype(float).reset_index(drop=True)
    blocks.append(controls)
    term_types.update({c: "weather_control" for c in WEATHER_CONTROLS})
    city = pd.get_dummies(f["city_key"].astype(str), prefix="city_FE", drop_first=True, dtype=float).reset_index(drop=True)
    year = pd.get_dummies(f["year"].astype(int), prefix="year_FE", drop_first=True, dtype=float).reset_index(drop=True)
    blocks.extend([city, year])
    term_types.update({c: "city_fixed_effect" for c in city.columns})
    term_types.update({c: "year_fixed_effect" for c in year.columns})
    design = pd.concat(blocks, axis=1)
    names = design.columns.astype(str).tolist()
    X = design.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(X))
    meta = {
        "rank": rank,
        "dimension": int(X.shape[1]),
        "city_count": int(f["city_key"].nunique()),
        "year_count": int(f["year"].nunique()),
        "city_reference": str(sorted(f["city_key"].astype(str).unique())[0]),
        "year_reference": int(sorted(f["year"].astype(int).unique())[0]),
        "city_levels": sorted(f["city_key"].astype(str).unique().tolist()),
        "year_levels": sorted(f["year"].astype(int).unique().tolist()),
    }
    if rank != X.shape[1]:
        raise np.linalg.LinAlgError(f"design rank deficient before fit: rank={rank}, dim={X.shape[1]}")
    return X, names, term_types, meta


def cluster_cr1_covariance(
    scores: np.ndarray, clusters: pd.Series, bread: np.ndarray, n: int, rank: int
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = clusters.astype(str).to_numpy()
    unique = pd.unique(labels)
    meat = np.zeros((scores.shape[1], scores.shape[1]), dtype=float)
    for label in unique:
        summed = scores[labels == label].sum(axis=0)
        meat += np.outer(summed, summed)
    g = int(len(unique))
    if g <= 1 or n <= rank:
        raise ValueError(f"invalid CR1 dimensions: n={n}, rank={rank}, G={g}")
    factor = (g / (g - 1.0)) * ((n - 1.0) / (n - rank))
    covariance = factor * bread @ meat @ bread
    covariance = (covariance + covariance.T) / 2.0
    return covariance, {
        "cluster_level": "city_key",
        "clusters": g,
        "cr1_factor": float(factor),
        "cr1_formula": "(G/(G-1))*((n-1)/(n-rank))",
    }


def _history_rows(model_id: str, fit_history: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten statsmodels' numeric history without storing parameter vectors."""

    rows: list[dict[str, Any]] = []
    deviance = fit_history.get("deviance", []) if isinstance(fit_history, Mapping) else []
    params = fit_history.get("params", []) if isinstance(fit_history, Mapping) else []
    if not isinstance(deviance, (list, tuple, np.ndarray)):
        deviance = []
    if not isinstance(params, (list, tuple, np.ndarray)):
        params = []
    count = max(len(deviance), len(params))
    for i in range(count):
        d = deviance[i] if i < len(deviance) else np.nan
        p = params[i] if i < len(params) else np.nan
        p_arr = np.asarray(p, dtype=float) if np.ndim(p) else np.asarray([p], dtype=float)
        rows.append(
            {
                "model_id": model_id,
                "diagnostic_type": "irls_iteration",
                "iteration": int(i),
                "deviance": float(d) if np.isfinite(d) else np.nan,
                "parameter_l2": float(np.linalg.norm(p_arr)) if np.isfinite(p_arr).all() else np.nan,
            }
        )
    return rows


def fit_ppml(
    model_id: str,
    frame: pd.DataFrame,
    X: np.ndarray,
    names: Sequence[str],
    term_types: Mapping[str, str],
    design_meta: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit one model, with at most one logged equivalent solver repair."""

    y = frame["Q"].to_numpy(float)
    offset = frame["lnW"].to_numpy(float)
    n_obs, p = X.shape
    rank = int(np.linalg.matrix_rank(X))
    clusters = int(frame["city_key"].nunique())
    base: dict[str, Any] = {
        "model_id": model_id,
        "method": "Poisson QMLE",
        "outcome": "Q",
        "n": int(n_obs),
        "cities": clusters,
        "rank": rank,
        "dimension": int(p),
        "df_city_cluster": int(clusters - 1),
        "offset": "lnW",
        "covariance_method": "Poisson score/Hessian city-clustered CR1 sandwich; t(G-1)",
        "fit_status": "failed",
        "converged": False,
        "iterations": 0,
        "fit_attempts": 0,
        "numeric_repair_used": False,
        "failure_reason": "",
        "warnings": [],
        "score_max_abs": np.nan,
        "score_l2": np.nan,
        "residual_mean": np.nan,
        "residual_min": np.nan,
        "residual_max": np.nan,
        "fitted_min": np.nan,
        "fitted_max": np.nan,
        "hessian_condition_number": np.nan,
        "loglik": np.nan,
        "deviance": np.nan,
        "pearson_chi2": np.nan,
        "beta_lnN": np.nan,
        "se_lnN": np.nan,
        "ci_low_lnN": np.nan,
        "ci_high_lnN": np.nan,
        "p_value_lnN": np.nan,
        "critical_t": np.nan,
        "covariance": np.full((p, p), np.nan),
        "beta": np.full(p, np.nan),
        "se": np.full(p, np.nan),
        "ci_low": np.full(p, np.nan),
        "ci_high": np.full(p, np.nan),
        "p_value": np.full(p, np.nan),
        "names": list(names),
        "term_types": dict(term_types),
        "design_meta": dict(design_meta),
    }
    numeric_rows: list[dict[str, Any]] = []
    if rank != p:
        base["failure_reason"] = f"design rank deficient: rank={rank}, dim={p}"
        numeric_rows.append({"model_id": model_id, "diagnostic_type": "pre_fit", "status": "failed", "failure_reason": base["failure_reason"]})
        return base, numeric_rows

    # The first call is the existing IRLS implementation used by the trusted
    # V12 PPML.  A single Newton call is allowed only if IRLS raises or fails
    # to converge; it is not a new model specification.
    attempts = [("IRLS", {"method": "IRLS", "maxiter": 500, "tol": 1e-10})]
    fit = None
    caught_all: list[str] = []
    for attempt_index, (method, kwargs) in enumerate(attempts):
        base["fit_attempts"] = attempt_index + 1
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit(disp=0, **kwargs)
            caught_all.extend(str(item.message) for item in caught)
            if bool(getattr(fit, "converged", False)) and np.isfinite(np.asarray(fit.params, dtype=float)).all():
                break
            if attempt_index == 0:
                # Do not silently switch methods.  The equivalent repair is
                # represented explicitly in the attempts list below only when
                # the primary IRLS result is unusable.
                attempts.append(("Newton_repair", {"method": "newton", "maxiter": 500, "tol": 1e-10}))
                base["numeric_repair_used"] = True
                continue
        except Exception as exc:
            caught_all.append(f"{method}: {type(exc).__name__}: {exc}")
            if attempt_index == 0:
                attempts.append(("Newton_repair", {"method": "newton", "maxiter": 500, "tol": 1e-10}))
                base["numeric_repair_used"] = True
                continue
        break

    base["warnings"] = caught_all
    if fit is None:
        base["failure_reason"] = "; ".join(caught_all[-3:]) or "PPML fit returned no result"
        numeric_rows.append({"model_id": model_id, "diagnostic_type": "fit", "status": "failed", "failure_reason": base["failure_reason"], "fit_attempts": base["fit_attempts"]})
        return base, numeric_rows

    beta = np.asarray(fit.params, dtype=float)
    base["converged"] = bool(getattr(fit, "converged", False))
    base["iterations"] = int(fit.fit_history.get("iteration", 0)) if isinstance(getattr(fit, "fit_history", None), dict) else 0
    base["loglik"] = float(getattr(fit, "llf", np.nan))
    base["deviance"] = float(getattr(fit, "deviance", np.nan))
    base["pearson_chi2"] = float(getattr(fit, "pearson_chi2", np.nan))
    numeric_rows.extend(_history_rows(model_id, getattr(fit, "fit_history", {})))
    if not base["converged"]:
        base["failure_reason"] = f"GLM did not converge after {base['iterations']} iterations"
    if not np.isfinite(beta).all():
        base["failure_reason"] = "non-finite PPML coefficient"

    # Keep a finite result only when the fitted mean and the requested
    # Poisson score/Hessian covariance are usable.  This avoids reporting a
    # coefficient while silently using a singular or non-finite covariance.
    try:
        mu = np.exp(np.clip(offset + X @ beta, -745.0, 709.0))
        if not np.isfinite(mu).all():
            raise FloatingPointError("non-finite PPML fitted mean")
        hessian = X.T @ (mu[:, None] * X)
        hessian_condition = float(np.linalg.cond(hessian))
        bread = np.linalg.inv(hessian)
        scores = X * (y - mu)[:, None]
        covariance, cov_meta = cluster_cr1_covariance(scores, frame["city_key"], bread, n_obs, rank)
        se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        df = clusters - 1
        critical = float(stats.t.ppf(0.975, df))
        ci_low = beta - critical * se
        ci_high = beta + critical * se
        t_value = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
        p_value = 2.0 * stats.t.sf(np.abs(t_value), df)
        total_score = scores.sum(axis=0)
        base.update(
            {
                "covariance": covariance,
                "beta": beta,
                "se": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "critical_t": critical,
                "score_max_abs": float(np.max(np.abs(total_score))),
                "score_l2": float(np.linalg.norm(total_score)),
                "residual_mean": float(np.mean(y - mu)),
                "residual_min": float(np.min(y - mu)),
                "residual_max": float(np.max(y - mu)),
                "fitted_min": float(np.min(mu)),
                "fitted_max": float(np.max(mu)),
                "hessian_condition_number": hessian_condition,
                "beta_lnN": float(beta[list(names).index("lnN")]),
                "se_lnN": float(se[list(names).index("lnN")]),
                "ci_low_lnN": float(ci_low[list(names).index("lnN")]),
                "ci_high_lnN": float(ci_high[list(names).index("lnN")]),
                "p_value_lnN": float(p_value[list(names).index("lnN")]),
                "covariance_meta": cov_meta,
            }
        )
        if base["converged"] and np.isfinite(beta).all() and np.isfinite(covariance).all():
            base["fit_status"] = "estimated"
        elif np.isfinite(beta).all() and np.isfinite(covariance).all():
            base["fit_status"] = "not_converged"
        else:
            base["fit_status"] = "failed"
    except Exception as exc:
        base["fit_status"] = "failed"
        base["failure_reason"] = f"{type(exc).__name__}: {exc}"

    numeric_rows.append(
        {
            "model_id": model_id,
            "diagnostic_type": "fit_summary",
            "status": base["fit_status"],
            "converged": base["converged"],
            "fit_attempts": base["fit_attempts"],
            "numeric_repair_used": base["numeric_repair_used"],
            "iterations": base["iterations"],
            "rank": base["rank"],
            "dimension": base["dimension"],
            "clusters": base["cities"],
            "score_max_abs": base["score_max_abs"],
            "score_l2": base["score_l2"],
            "residual_mean": base["residual_mean"],
            "residual_min": base["residual_min"],
            "residual_max": base["residual_max"],
            "fitted_min": base["fitted_min"],
            "fitted_max": base["fitted_max"],
            "hessian_condition_number": base["hessian_condition_number"],
            "failure_reason": base["failure_reason"],
        }
    )
    return base, numeric_rows


def specs_table(
    panel: pd.DataFrame,
    info: Mapping[str, Any],
    design_meta: Mapping[str, Mapping[str, Any]],
    sample_signature: str,
) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    model_frames: Mapping[str, pd.DataFrame] = info["model_frames"]
    for model_id in MODEL_IDS:
        frame = model_frames[model_id]
        positive = int(frame["Q"].gt(0).sum())
        zeros = int(frame["Q"].eq(0).sum())
        rows.append(
            {
                "model_id": model_id,
                "window": "annual 2016-2023",
                "method": "Poisson QMLE",
                "outcome": "Q",
                "formula": MODEL_FORMULA,
                "offset": "lnW=ln(W), exposure only; W is not frequency weight",
                "fixed_effects": "city_key FE + year FE",
                "elasticity_regressor": "lnN=ln(E/W)",
                "weather_controls": "+".join(WEATHER_CONTROLS),
                "candidate_rule": SAMPLE_RULE_ZERO if model_id == "P0_with_zeros" else "Q>0; E>0; W>0; finite positive N=E/W; five weather controls complete",
                "all_zero_city_rule": "remove candidate cities with Q sum = 0 before singleton rule" if model_id == "P0_with_zeros" else "not applicable to positive-Q candidate",
                "singleton_rule": "remove candidate cities with exactly one row after preceding rule",
                "candidate_rows": int(info["candidate_rows"] if model_id == "P0_with_zeros" else info["positive_candidate_rows"]),
                "candidate_cities": int(info["candidate_cities"] if model_id == "P0_with_zeros" else info["positive_candidate_cities"]),
                "candidate_positive_rows": int(info["candidate_positive_rows"] if model_id == "P0_with_zeros" else info["positive_candidate_rows"]),
                "candidate_zero_rows": int(info["candidate_zero_rows"] if model_id == "P0_with_zeros" else 0),
                "all_zero_cities_removed": int(info["all_zero_city_count"] if model_id == "P0_with_zeros" else 0),
                "all_zero_rows_removed": int(info["all_zero_rows"] if model_id == "P0_with_zeros" else 0),
                "singleton_cities_removed": int(info["singleton_zero_city_count"] if model_id == "P0_with_zeros" else info["positive_singleton_city_count"]),
                "singleton_rows_removed": int(info["singleton_zero_rows"] if model_id == "P0_with_zeros" else info["positive_singleton_rows"]),
                "model_rows": int(len(frame)),
                "model_cities": int(frame["city_key"].nunique()),
                "model_positive_rows": positive,
                "model_zero_rows": zeros,
                "design_dimension_prefit": int(design_meta[model_id]["dimension"]),
                "design_rank_prefit": int(design_meta[model_id]["rank"]),
                "design_year_count": int(design_meta[model_id]["year_count"]),
                "city_reference": design_meta[model_id]["city_reference"],
                "year_reference": int(design_meta[model_id]["year_reference"]),
                "sample_signature": sample_signature,
                "prefit_status": "frozen_before_fit",
                "freeze_utc": utc_now(),
            }
        )
    specs = pd.DataFrame(rows)
    signature = hashlib.sha256(specs.to_csv(index=False).encode("utf-8")).hexdigest()
    return specs, signature


def sample_signature(panel: pd.DataFrame, info: Mapping[str, Any]) -> str:
    model_frames: Mapping[str, pd.DataFrame] = info["model_frames"]
    payload: dict[str, Any] = {
        "candidate": panel.loc[panel["candidate_preflight"], ["analysis_row_id", "city_key", "year", "Q", "E", "W", *WEATHER_CONTROLS]].sort_values(["city_key", "year"]).round(12).to_dict(orient="records"),
        "models": {
            key: value[["analysis_row_id", "city_key", "year", "Q", "E", "W"]].sort_values(["city_key", "year"]).to_dict(orient="records")
            for key, value in model_frames.items()
        },
        "controls": list(WEATHER_CONTROLS),
        "rule": SAMPLE_RULE_ZERO,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def coefficients_table(fits: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        fit = fits[model_id]
        for i, term in enumerate(fit["names"]):
            rows.append(
                {
                    "model_id": model_id,
                    "term": term,
                    "term_type": fit["term_types"].get(term, "other"),
                    "coefficient": fit["beta"][i] if np.isfinite(fit["beta"][i]) else np.nan,
                    "se": fit["se"][i] if np.isfinite(fit["se"][i]) else np.nan,
                    "ci_low": fit["ci_low"][i] if np.isfinite(fit["ci_low"][i]) else np.nan,
                    "ci_high": fit["ci_high"][i] if np.isfinite(fit["ci_high"][i]) else np.nan,
                    "p_value": fit["p_value"][i] if np.isfinite(fit["p_value"][i]) else np.nan,
                    "n": fit["n"],
                    "cities": fit["cities"],
                    "rank": fit["rank"],
                    "dimension": fit["dimension"],
                    "df_city_cluster": fit["df_city_cluster"],
                    "fit_status": fit["fit_status"],
                    "failure_reason": fit["failure_reason"],
                    "covariance_method": fit["covariance_method"],
                    "offset": fit["offset"],
                }
            )
    return pd.DataFrame(rows)


def covariance_table(fits: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        fit = fits[model_id]
        covariance = fit["covariance"]
        for i, term_i in enumerate(fit["names"]):
            for j, term_j in enumerate(fit["names"]):
                value = covariance[i, j]
                rows.append(
                    {
                        "model_id": model_id,
                        "term_i": term_i,
                        "term_j": term_j,
                        "covariance": value if np.isfinite(value) else np.nan,
                        "fit_status": fit["fit_status"],
                    }
                )
    return pd.DataFrame(rows)


def results_table(fits: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        fit = fits[model_id]
        rows.append(
            {
                "model_id": model_id,
                "method": fit["method"],
                "outcome": fit["outcome"],
                "parameter": "lnN",
                # Generic aliases keep the result interface compatible with
                # the V15 figure/manuscript readers; the explicit *_lnN
                # columns below remain the auditable canonical names.
                "beta": fit["beta_lnN"],
                "se": fit["se_lnN"],
                "ci_low": fit["ci_low_lnN"],
                "ci_high": fit["ci_high_lnN"],
                "p_value": fit["p_value_lnN"],
                "beta_lnN": fit["beta_lnN"],
                "se_lnN": fit["se_lnN"],
                "ci_low_lnN": fit["ci_low_lnN"],
                "ci_high_lnN": fit["ci_high_lnN"],
                "p_value_lnN": fit["p_value_lnN"],
                "n": fit["n"],
                "cities": fit["cities"],
                "rank": fit["rank"],
                "dimension": fit["dimension"],
                "df_city_cluster": fit["df_city_cluster"],
                "critical_t": fit["critical_t"],
                "fit_status": fit["fit_status"],
                "converged": fit["converged"],
                "iterations": fit["iterations"],
                "fit_attempts": fit["fit_attempts"],
                "numeric_repair_used": fit["numeric_repair_used"],
                "score_max_abs": fit["score_max_abs"],
                "score_l2": fit["score_l2"],
                "residual_mean": fit["residual_mean"],
                "residual_min": fit["residual_min"],
                "residual_max": fit["residual_max"],
                "fitted_min": fit["fitted_min"],
                "fitted_max": fit["fitted_max"],
                "hessian_condition_number": fit["hessian_condition_number"],
                "loglik": fit["loglik"],
                "deviance": fit["deviance"],
                "pearson_chi2": fit["pearson_chi2"],
                "offset": fit["offset"],
                "covariance_method": fit["covariance_method"],
                "failure_reason": fit["failure_reason"],
            }
        )
    return pd.DataFrame(rows)


def output_hashes() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(OUTPUT.glob("ppml_zero_*.csv")):
        if path.is_file():
            result[path.name] = {"path": rel(path), "bytes": int(path.stat().st_size), "sha256": sha256(path)}
    return result


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    start = utc_now()
    panel, candidate, info = build_samples()
    models: Mapping[str, pd.DataFrame] = info["model_frames"]
    design: dict[str, tuple[np.ndarray, list[str], dict[str, str], dict[str, Any]]] = {}
    for model_id in MODEL_IDS:
        design[model_id] = make_design(models[model_id])
    design_meta = {model_id: value[3] for model_id, value in design.items()}
    sample_sig = sample_signature(panel, info)
    specs, spec_sig = specs_table(panel, info, design_meta, sample_sig)

    # Freeze the specifications and full audit table before any estimator call.
    specs_path = OUTPUT / "ppml_zero_specs_v15.csv"
    sample_path = OUTPUT / "ppml_zero_sample_v15.csv"
    specs.to_csv(specs_path, index=False, encoding="utf-8-sig", float_format="%.15g")
    panel_out_cols = [
        "analysis_row_id", "city_key", "city", "province", "year", "source_city", "Q", "E", "W", "N", "lnN", "lnW",
        "q_status", "e_status", "w_status", "q_zero_semantics", "e_zero_semantics", "service_status",
        *WEATHER_CONTROLS, "weather_key_match", "q_nonmissing", "q_nonnegative", "w_positive", "e_positive", "n_finite_positive",
        "weather_complete", "candidate_preflight", "source_encoded_zero", "candidate_q_state", "preflight_exclusion_reason",
        "candidate_all_zero_city", "after_all_zero_removal", "p0_with_zeros_singleton_city", "p0_with_zeros_model_member",
        "p0_with_zeros_sample_role", "p0_with_zeros_exclusion_reason", "p0_positive_candidate", "p0_positive_singleton_city",
        "p0_positive_model_member", "p0_positive_sample_role", "p0_positive_exclusion_reason", "sample_rule_version",
    ]
    panel_out_cols = [c for c in panel_out_cols if c in panel.columns]
    panel[panel_out_cols].to_csv(sample_path, index=False, encoding="utf-8-sig", float_format="%.15g")

    fits: dict[str, dict[str, Any]] = {}
    numeric_rows: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        X, names, term_types, meta = design[model_id]
        fit, diagnostics = fit_ppml(model_id, models[model_id], X, names, term_types, meta)
        fits[model_id] = fit
        numeric_rows.extend(diagnostics)
        fit_records.append(
            {
                "model_id": model_id,
                "fit_status": fit["fit_status"],
                "converged": fit["converged"],
                "iterations": fit["iterations"],
                "fit_attempts": fit["fit_attempts"],
                "numeric_repair_used": fit["numeric_repair_used"],
                "n": fit["n"],
                "cities": fit["cities"],
                "rank": fit["rank"],
                "dimension": fit["dimension"],
                "failure_reason": fit["failure_reason"],
                "utc": utc_now(),
            }
        )

    results = results_table(fits)
    coefficients = coefficients_table(fits)
    covariance = covariance_table(fits)
    numeric_log = pd.DataFrame(numeric_rows)
    results.to_csv(OUTPUT / "ppml_zero_results_v15.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    coefficients.to_csv(OUTPUT / "ppml_zero_coefficients_v15.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    covariance.to_csv(OUTPUT / "ppml_zero_covariance_v15.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    numeric_log.to_csv(OUTPUT / "ppml_zero_numeric_log_v15.csv", index=False, encoding="utf-8-sig", float_format="%.15g")

    inputs = {
        "annual_service_metrics_v9": {"path": rel(ANNUAL_INPUT), "sha256": sha256(ANNUAL_INPUT)},
        "weather_city_year_v4": {"path": rel(WEATHER_INPUT), "sha256": sha256(WEATHER_INPUT)},
        "script": {"path": rel(HERE), "sha256": sha256(HERE)},
    }
    model_status: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        fit = fits[model_id]
        model_status[model_id] = {
            key: fit.get(key)
            for key in (
                "fit_status", "converged", "iterations", "fit_attempts", "numeric_repair_used", "n", "cities", "rank", "dimension",
                "df_city_cluster", "critical_t", "beta_lnN", "se_lnN", "ci_low_lnN", "ci_high_lnN", "p_value_lnN",
                "score_max_abs", "score_l2", "residual_mean", "residual_min", "residual_max", "fitted_min", "fitted_max",
                "hessian_condition_number", "loglik", "deviance", "pearson_chi2", "failure_reason", "warnings", "covariance_meta",
            )
        }
    status = {
        "script": rel(HERE),
        "script_version": SCRIPT_VERSION,
        "started_utc": start,
        "finished_utc": utc_now(),
        "fit_mode": "fresh_fit_two_prespecified_specs",
        "models_planned": list(MODEL_IDS),
        "models_fitted": len(MODEL_IDS),
        "actual_model_fit_calls": int(sum(f["fit_attempts"] for f in fits.values())),
        "primary_model_fit_calls": len(MODEL_IDS),
        "numeric_equivalent_repair_attempts": int(sum(bool(f["numeric_repair_used"]) for f in fits.values())),
        "spec_signature": spec_sig,
        "sample_signature": sample_sig,
        "inputs": inputs,
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "sample": {
            key: value
            for key, value in info.items()
            if key not in {"model_frames", "candidate_city_keys", "all_zero_city_keys", "singleton_zero_city_keys", "positive_singleton_city_keys"}
        },
        "sample_keys": {
            "all_zero_city_keys": sorted(info["all_zero_city_keys"]),
            "singleton_zero_city_keys": sorted(info["singleton_zero_city_keys"]),
            "positive_singleton_city_keys": sorted(info["positive_singleton_city_keys"]),
        },
        "models": model_status,
        "inference": {
            "score": "X_i*(Q_i-mu_i), mu_i=exp(lnW_i+X_i beta)",
            "hessian": "X' diag(mu) X",
            "cluster_level": "city_key",
            "cr1": "(G/(G-1))*((n-1)/(n-rank))",
            "interval": "beta +/- t_(0.975,G-1)*SE; city-cluster CR1",
            "offset": "lnW; W is exposure only, not frequency weight",
        },
        "fit_records": fit_records,
        "outputs": output_hashes(),
    }
    write_json(OUTPUT / "ppml_zero_status_v15.json", status)
    # The status hash is intentionally not included in itself.  Refresh the
    # output inventory after writing it for the human-readable run summary.
    status["outputs"] = output_hashes()
    write_json(OUTPUT / "ppml_zero_status_v15.json", status)
    return status


if __name__ == "__main__":
    summary = main()
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))
