"""Fit the annual fixed-elasticity, restricted-cubic-spline, and PPML models.

The public entry point is intentionally self-contained. It reads the shipped
annual service metrics and weather table, constructs the 464-row
positive Q/E/W candidate, freezes the common multi-period sample and the four
knots, and only then calls the three estimators. Source inputs are never
written. The script writes the nonlinear result tables and keeps a
signature-based cache so a repeat with unchanged inputs does not refit a model.

The spline is an explicit two-term restricted cubic spline, not a data-selected
GAM basis. L0/L1 use OLS with city-clustered CR1 t
inference.  P1 uses Poisson QMLE with log(W) as an offset and a score/Hessian
city-clustered sandwich covariance; W is not also used as a frequency weight.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


HERE = Path(__file__).resolve()
REPOSITORY = HERE.parents[2]
OUTPUT = REPOSITORY / "data" / "results" / "recomputed_annual"
OUTPUT.mkdir(parents=True, exist_ok=True)
ANNUAL_INPUT = REPOSITORY / "data" / "processed" / "annual_service_metrics_v9.csv"
WEATHER_INPUT = REPOSITORY / "data" / "processed" / "weather_city_year_v4.csv"

MODEL_IDS = ("L0", "L1", "P1")
PRIMARY_WEATHER = (
    "weather_hdd18_degree_days",
    "weather_cdd24_degree_days",
    "weather_precipitation_sum_mm",
    "weather_wind_speed_mean_ms",
    "weather_relative_humidity_mean_pct",
)
FALLBACK_WEATHER = (
    "weather_temperature_mean_c",
    "weather_precipitation_sum_mm",
    "weather_wind_speed_mean_ms",
    "weather_relative_humidity_mean_pct",
)
OUTCOMES = {"L0": "lnA", "L1": "lnA", "P1": "Q"}
QUANTILES = (0.05, 0.35, 0.65, 0.95)
GRID_SIZE = 201
SCRIPT_VERSION = "v12_explicit_rcs_ols_ppml_score_hessian_2026-11-04"


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


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
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
        json.dumps(json_safe(dict(payload)), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def positive_part(value: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(value, dtype=float), 0.0)


def rcs_basis(x: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    """Return the two explicit restricted-cubic basis columns.

    The first two entries of the ordered knot vector are the interior basis
    knots; the final two entries are the boundary knots.
    """

    values = np.asarray(x, dtype=float)
    k1, k2, k3, k4 = map(float, knots)
    denominator = (k4 - k1) ** 2
    if denominator <= 0 or k4 <= k3:
        raise ValueError(f"Invalid restricted-cubic knot spacing: {knots}")
    out = []
    for kj in (k1, k2):
        out.append(
            (
                positive_part(values - kj) ** 3
                - (k4 - kj) / (k4 - k3) * positive_part(values - k3) ** 3
                + (k3 - kj) / (k4 - k3) * positive_part(values - k4) ** 3
            )
            / denominator
        )
    return np.column_stack(out)


def rcs_derivative(x: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    """Analytic d(H1,H2)/dx for the explicit restricted cubic basis."""

    values = np.asarray(x, dtype=float)
    k1, k2, k3, k4 = map(float, knots)
    denominator = (k4 - k1) ** 2
    if denominator <= 0 or k4 <= k3:
        raise ValueError(f"Invalid restricted-cubic knot spacing: {knots}")
    out = []
    for kj in (k1, k2):
        out.append(
            (
                3.0 * positive_part(values - kj) ** 2
                - 3.0 * (k4 - kj) / (k4 - k3) * positive_part(values - k3) ** 2
                + 3.0 * (k3 - kj) / (k4 - k3) * positive_part(values - k4) ** 2
            )
            / denominator
        )
    return np.column_stack(out)


def basis_formula() -> str:
    return (
        "Hj(x)=((x-kj)_+^3-(k4-kj)/(k4-k3)*(x-k3)_+^3+"
        "(k3-kj)/(k4-k3)*(x-k4)_+^3)/(k4-k1)^2; j=1,2"
    )


def derivative_formula() -> str:
    return (
        "Hj'(x)=(3(x-kj)_+^2-3*(k4-kj)/(k4-k3)*(x-k3)_+^2+"
        "3*(k3-kj)/(k4-k3)*(x-k4)_+^2)/(k4-k1)^2; j=1,2"
    )


def choose_controls(weather: pd.DataFrame) -> tuple[list[str], str]:
    if all(c in weather.columns for c in PRIMARY_WEATHER):
        return list(PRIMARY_WEATHER), "V4 HDD18/CDD24 plus precipitation, wind speed, relative humidity"
    if all(c in weather.columns for c in FALLBACK_WEATHER):
        return list(FALLBACK_WEATHER), "four-weather fallback: temperature, precipitation, wind speed, relative humidity"
    missing_primary = sorted(set(PRIMARY_WEATHER).difference(weather.columns))
    missing_fallback = sorted(set(FALLBACK_WEATHER).difference(weather.columns))
    raise KeyError(f"No complete common weather set; missing primary={missing_primary}; fallback={missing_fallback}")


def build_annual_sample() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the positive candidate and the common sample before estimation."""

    if not ANNUAL_INPUT.exists() or not WEATHER_INPUT.exists():
        raise FileNotFoundError(f"Missing annual nonlinear inputs: {ANNUAL_INPUT}, {WEATHER_INPUT}")
    annual = read_csv(ANNUAL_INPUT)
    weather = read_csv(WEATHER_INPUT)
    required = {"city_key", "city", "province", "year", "Q", "E", "W"}
    missing = sorted(required.difference(annual.columns))
    if missing:
        raise KeyError(f"annual_service_metrics_v9 missing columns: {missing}")
    if weather.duplicated(["city_key", "year"]).any():
        raise ValueError("weather_city_year_v4 is not unique on city_key-year")
    controls, controls_source = choose_controls(weather)

    q = numeric(annual["Q"])
    e = numeric(annual["E"])
    w = numeric(annual["W"])
    positive = q.gt(0) & e.gt(0) & w.gt(0)
    candidate = annual.loc[positive].copy()
    if len(candidate) != 464 or candidate["city_key"].nunique() != 123:
        raise ValueError(
            f"Expected annual positive candidate changed: rows={len(candidate)}, cities={candidate['city_key'].nunique()}"
        )
    candidate = candidate[[c for c in ["analysis_row_id", "city_key", "city", "province", "year", "Q", "E", "W", "I"] if c in candidate.columns]].copy()
    candidate["Q"] = numeric(candidate["Q"])
    candidate["E"] = numeric(candidate["E"])
    candidate["W"] = numeric(candidate["W"])
    candidate["year"] = numeric(candidate["year"]).astype(int)
    candidate["A"] = candidate["Q"] / candidate["W"]
    candidate["N"] = candidate["E"] / candidate["W"]
    candidate["I"] = candidate["Q"] / candidate["E"]
    candidate["lnA"] = np.log(candidate["A"])
    candidate["lnN"] = np.log(candidate["N"])
    candidate["lnI"] = np.log(candidate["I"])
    # Q/E/W and their derived ratios are the common definitions.  Existing I
    # is retained only as a source diagnostic, never as a replacement.
    if "I" in annual.columns:
        source_i = numeric(annual.loc[positive, "I"]).to_numpy(float)
        identity_err = np.nanmax(np.abs(source_i - candidate["I"].to_numpy(float)))
        if np.isfinite(identity_err) and identity_err > 1e-8 * (1 + np.nanmax(np.abs(candidate["I"]))):
            raise ValueError(f"Annual I source is inconsistent with Q/E: max_abs_error={identity_err}")

    merged = candidate.merge(
        weather[["city_key", "year", *controls]],
        on=["city_key", "year"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing_rows = merged.loc[merged["_merge"].ne("both"), ["city_key", "year"]]
        raise ValueError(f"Annual candidate missing weather rows: {missing_rows.head().to_dict('records')}")
    merged.drop(columns=["_merge"], inplace=True)
    for c in controls:
        merged[c] = numeric(merged[c])
    merged["common_controls_available"] = merged[controls].notna().all(axis=1)
    # A city with one positive compatible year contains no within-city signal
    # for a city FE slope. The fixed sample rule excludes those rows once, before
    # all three models, while retaining all other positive extremes.
    city_counts = merged.groupby("city_key", sort=False).size()
    merged["city_positive_years"] = merged["city_key"].map(city_counts).astype(int)
    merged["singleton_city"] = merged["city_positive_years"].eq(1)
    merged["model_member"] = merged["common_controls_available"] & ~merged["singleton_city"]
    merged["candidate_member"] = True
    merged["exclusion_reason"] = ""
    merged.loc[~merged["common_controls_available"], "exclusion_reason"] = "missing_common_weather_control"
    merged.loc[merged["singleton_city"] & merged["common_controls_available"], "exclusion_reason"] = "single_positive_year_city_no_within_city_information"
    merged["sample_role"] = np.where(merged["model_member"], "common_model_sample", "candidate_excluded")
    merged["x"] = merged["lnN"]
    merged.sort_values(["city_key", "year", "analysis_row_id"], inplace=True, ignore_index=True)
    model = merged.loc[merged["model_member"]].copy()
    if len(model) != 445 or model["city_key"].nunique() != 104:
        raise ValueError(f"Expected nonlinear model sample changed: rows={len(model)}, cities={model['city_key'].nunique()}")
    if model[controls].isna().any().any():
        raise ValueError("Common weather controls contain missing values after model membership freeze")
    knots = np.quantile(model["x"].to_numpy(float), QUANTILES, method="linear")
    if len(np.unique(knots)) != 4:
        raise ValueError(f"RCS knots are not unique: {knots.tolist()}")
    return merged, model, {
        "candidate_rows": int(len(merged)),
        "candidate_cities": int(merged["city_key"].nunique()),
        "candidate_years": sorted(merged["year"].unique().astype(int).tolist()),
        "candidate_positive_rule": "Q>0 and E>0 and W>0",
        "candidate_singleton_rows": int(merged["singleton_city"].sum()),
        "candidate_singleton_cities": int(merged.loc[merged["singleton_city"], "city_key"].nunique()),
        "model_rows": int(len(model)),
        "model_cities": int(model["city_key"].nunique()),
        "model_years": sorted(model["year"].unique().astype(int).tolist()),
        "controls": controls,
        "controls_source": controls_source,
        "controls_missing_rows_candidate": int((~merged["common_controls_available"]).sum()),
        "knots": [float(k) for k in knots],
        "quantile_levels": list(QUANTILES),
        "quantile_method": "numpy.quantile(method='linear')",
        "x_definition": "x=ln(N)=ln(E/W)",
        "outcome_definitions": {"A": "Q/W", "N": "E/W", "I": "Q/E", "lnA": "ln(Q/W)", "lnN": "ln(E/W)", "lnI": "ln(Q/E)"},
    }


def sample_signature(model: pd.DataFrame, controls: Sequence[str], knots: Sequence[float]) -> str:
    key_rows = model[["city_key", "year", "Q", "E", "W", *controls]].copy()
    key_rows = key_rows.sort_values(["city_key", "year"]).round(12)
    payload = {
        "rows": key_rows.to_dict(orient="records"),
        "controls": list(controls),
        "knots": [float(k) for k in knots],
        "sample_rule": "positive Q/E/W; common controls; exclude singleton city",
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def freeze_specs(info: Mapping[str, Any], sample_sig: str) -> tuple[Path, str]:
    knots = [float(k) for k in info["knots"]]
    core = {
        "sample_signature": sample_sig,
        "candidate_rows": info["candidate_rows"],
        "candidate_cities": info["candidate_cities"],
        "model_rows": info["model_rows"],
        "model_cities": info["model_cities"],
        "controls": list(info["controls"]),
        "controls_source": info["controls_source"],
        "knots": knots,
        "quantile_levels": list(QUANTILES),
        "quantile_method": info["quantile_method"],
        "basis_formula": basis_formula(),
        "derivative_formula": derivative_formula(),
        "fe": "city_key FE + year FE",
        "singleton_rule": "exclude all positive rows for cities with one positive year",
        "p1_offset": "ln(W), not frequency weight",
        "models": {
            "L0": "lnA ~ x + common_controls + city_FE + year_FE",
            "L1": "lnA ~ x + H1(x) + H2(x) + common_controls + city_FE + year_FE",
            "P1": "Q|X,W ~ Poisson(exp(lnW + city_FE + year_FE + x + H1(x) + H2(x) + common_controls))",
        },
        "inference": {
            "OLS": "city-cluster CR1; t(G-1)",
            "PPML": "Poisson score/Hessian city-cluster CR1 sandwich; t(G-1) for pointwise intervals",
        },
    }
    spec_sig = hashlib.sha256(json.dumps(core, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    path = OUTPUT / "nonlinear_specs_v12.csv"
    rows = []
    for model_id in MODEL_IDS:
        rows.append(
            {
                "model_id": model_id,
                "window": "annual 2016-2023 positive compatible rows",
                "method": "OLS" if model_id != "P1" else "Poisson QMLE",
                "outcome": OUTCOMES[model_id],
                "formula": core["models"][model_id],
                "candidate_rows": info["candidate_rows"],
                "candidate_cities": info["candidate_cities"],
                "singleton_rows_excluded": info["candidate_singleton_rows"],
                "singleton_cities_excluded": info["candidate_singleton_cities"],
                "model_rows": info["model_rows"],
                "model_cities": info["model_cities"],
                "control_set": "+".join(info["controls"]),
                "control_source": info["controls_source"],
                "k1_p05": knots[0],
                "k2_p35": knots[1],
                "k3_p65": knots[2],
                "k4_p95": knots[3],
                "quantile_method": info["quantile_method"],
                "basis_formula": basis_formula(),
                "derivative_formula": derivative_formula(),
                "fixed_effects": "city_key + year",
                "offset": "lnW" if model_id == "P1" else "",
                "inference": "city CR1 t" if model_id != "P1" else "Poisson score/Hessian city-cluster CR1",
                "sample_signature": sample_sig,
                "spec_signature": spec_sig,
                "prefit_status": "frozen_before_fit",
                "freeze_utc": utc_now(),
            }
        )
    frozen = pd.DataFrame(rows)
    if path.exists():
        old = read_csv(path)
        if "spec_signature" not in old.columns or set(old["spec_signature"].astype(str)) != {spec_sig}:
            raise ValueError("Existing nonlinear_specs_v12.csv does not match frozen sample/knots/specification")
    else:
        frozen.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")
    return path, spec_sig


def make_design(frame: pd.DataFrame, model_id: str, knots: Sequence[float], controls: Sequence[str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    frame = frame.sort_values(["city_key", "year", "analysis_row_id"], kind="stable").reset_index(drop=True)
    x = frame["x"].to_numpy(float)
    blocks: list[pd.DataFrame] = [pd.DataFrame({"const": np.ones(len(frame)), "x": x})]
    term_types: dict[str, str] = {"const": "intercept", "x": "scale_linear"}
    if model_id in {"L1", "P1"}:
        h = rcs_basis(x, knots)
        blocks.append(pd.DataFrame({"H1": h[:, 0], "H2": h[:, 1]}))
        term_types.update({"H1": "nonlinear_basis", "H2": "nonlinear_basis"})
    control_block = frame[list(controls)].astype(float).copy()
    blocks.append(control_block)
    term_types.update({c: "weather_control" for c in controls})
    city = pd.get_dummies(frame["city_key"].astype(str), prefix="city_FE", drop_first=True, dtype=float)
    year = pd.get_dummies(frame["year"].astype(int), prefix="year_FE", drop_first=True, dtype=float)
    blocks.extend([city, year])
    term_types.update({c: "city_fixed_effect" for c in city.columns})
    term_types.update({c: "year_fixed_effect" for c in year.columns})
    design = pd.concat(blocks, axis=1)
    names = design.columns.astype(str).tolist()
    X = design.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(X))
    if rank != X.shape[1]:
        raise np.linalg.LinAlgError(f"{model_id} design rank deficient: rank={rank}, dim={X.shape[1]}")
    indices = {name: i for i, name in enumerate(names)}
    meta = {
        "term_types": term_types,
        "indices": indices,
        "rank": rank,
        "dim": int(X.shape[1]),
        "city_levels": sorted(frame["city_key"].astype(str).unique().tolist()),
        "year_levels": sorted(frame["year"].astype(int).unique().tolist()),
    }
    return X, names, meta


def cluster_cr1_covariance(scores: np.ndarray, clusters: pd.Series, bread: np.ndarray, n: int, rank: int) -> tuple[np.ndarray, dict[str, Any]]:
    labels = clusters.astype(str).to_numpy()
    unique = pd.unique(labels)
    meat = np.zeros((scores.shape[1], scores.shape[1]), dtype=float)
    for label in unique:
        s = scores[labels == label].sum(axis=0)
        meat += np.outer(s, s)
    G = int(len(unique))
    if G <= 1 or n <= rank:
        raise ValueError(f"Invalid cluster covariance dimensions: n={n}, rank={rank}, G={G}")
    factor = (G / (G - 1.0)) * ((n - 1.0) / (n - rank))
    covariance = factor * bread @ meat @ bread
    covariance = (covariance + covariance.T) / 2.0
    return covariance, {
        "cluster_level": "city_key",
        "clusters": G,
        "cr1_factor": float(factor),
        "cr1_formula": "(G/(G-1))*((n-1)/(n-rank))",
    }


def fit_ols(frame: pd.DataFrame, X: np.ndarray, names: Sequence[str], meta: Mapping[str, Any]) -> dict[str, Any]:
    y = frame["lnA"].to_numpy(float)
    n, p = X.shape
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    rank = int(rank)
    if rank != p:
        raise np.linalg.LinAlgError(f"OLS rank deficient: rank={rank}, dim={p}")
    residuals = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    scores = X * residuals[:, None]
    covariance, cov_meta = cluster_cr1_covariance(scores, frame["city_key"], bread, n, rank)
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    G = int(frame["city_key"].nunique())
    df = G - 1
    critical = float(stats.t.ppf(0.975, df))
    t_value = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_value = 2.0 * stats.t.sf(np.abs(t_value), df)
    ci_low = beta - critical * se
    ci_high = beta + critical * se
    rss = float(np.dot(residuals, residuals))
    tss = float(np.dot(y - np.mean(y), y - np.mean(y)))
    return {
        "model_id": meta["model_id"],
        "method": "OLS",
        "outcome": "lnA",
        "beta": beta,
        "covariance": covariance,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "names": list(names),
        "term_types": meta["term_types"],
        "n": n,
        "G": G,
        "rank": rank,
        "dim": p,
        "df": df,
        "critical": critical,
        "rss": rss,
        "r_squared": float(1.0 - rss / tss) if tss > 0 else np.nan,
        "converged": True,
        "iterations": 1,
        "status": "estimated",
        "failure_reason": "",
        "covariance_method": "OLS city-clustered CR1 sandwich; t(G-1)",
        "covariance_meta": cov_meta,
        "residuals": residuals,
        "fitted": X @ beta,
        "offset": "",
        "loglik": np.nan,
        "deviance": np.nan,
    }


def fit_ppml(frame: pd.DataFrame, X: np.ndarray, names: Sequence[str], meta: Mapping[str, Any]) -> dict[str, Any]:
    y = frame["Q"].to_numpy(float)
    offset = np.log(frame["W"].to_numpy(float))
    n, p = X.shape
    status = "estimated"
    reason = ""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit(
                maxiter=500,
                tol=1e-10,
                method="IRLS",
                disp=0,
            )
        beta = np.asarray(fit.params, dtype=float)
        if not bool(getattr(fit, "converged", False)):
            status = "not_converged"
            reason = f"statsmodels GLM IRLS did not converge after {fit.fit_history.get('iteration', 'unknown')} iterations"
        if not np.isfinite(beta).all():
            status = "failed"
            reason = "non-finite PPML coefficient"
        mu = np.exp(np.clip(offset + X @ beta, -745.0, 709.0))
        if not np.isfinite(mu).all():
            status = "failed"
            reason = "non-finite PPML fitted mean"
        # The covariance below is deliberately not fit.cov_params(): it is the
        # requested Poisson score/Hessian cluster sandwich.
        hessian = X.T @ (mu[:, None] * X)
        rank = int(np.linalg.matrix_rank(X))
        if rank != p:
            raise np.linalg.LinAlgError(f"PPML design rank deficient: rank={rank}, dim={p}")
        bread = np.linalg.inv(hessian)
        scores = X * (y - mu)[:, None]
        covariance, cov_meta = cluster_cr1_covariance(scores, frame["city_key"], bread, n, rank)
        se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        G = int(frame["city_key"].nunique())
        df = G - 1
        critical = float(stats.t.ppf(0.975, df))
        t_value = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
        p_value = 2.0 * stats.t.sf(np.abs(t_value), df)
        ci_low = beta - critical * se
        ci_high = beta + critical * se
        iterations = int(fit.fit_history.get("iteration", 0)) if isinstance(fit.fit_history, dict) else 0
        warnings_seen = [str(item.message) for item in caught]
        return {
            "model_id": meta["model_id"],
            "method": "Poisson QMLE",
            "outcome": "Q",
            "beta": beta,
            "covariance": covariance,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
            "names": list(names),
            "term_types": meta["term_types"],
            "n": n,
            "G": G,
            "rank": rank,
            "dim": p,
            "df": df,
            "critical": critical,
            "rss": np.nan,
            "r_squared": np.nan,
            "converged": bool(getattr(fit, "converged", False)),
            "iterations": iterations,
            "status": status,
            "failure_reason": reason,
            "covariance_method": "Poisson QMLE score/Hessian city-clustered CR1 sandwich; t(G-1)",
            "covariance_meta": {**cov_meta, "hessian": "X' diag(mu) X", "score": "X_i*(Q_i-mu_i)", "warnings": warnings_seen},
            "residuals": y - mu,
            "fitted": mu,
            "offset": "lnW",
            "loglik": float(getattr(fit, "llf", np.nan)),
            "deviance": float(getattr(fit, "deviance", np.nan)),
            "pearson_chi2": float(getattr(fit, "pearson_chi2", np.nan)),
        }
    except Exception as exc:
        return {
            "model_id": meta["model_id"],
            "method": "Poisson QMLE",
            "outcome": "Q",
            "beta": np.full(p, np.nan),
            "covariance": np.full((p, p), np.nan),
            "se": np.full(p, np.nan),
            "ci_low": np.full(p, np.nan),
            "ci_high": np.full(p, np.nan),
            "p_value": np.full(p, np.nan),
            "names": list(names),
            "term_types": meta["term_types"],
            "n": n,
            "G": int(frame["city_key"].nunique()),
            "rank": np.nan,
            "dim": p,
            "df": int(frame["city_key"].nunique() - 1),
            "critical": np.nan,
            "rss": np.nan,
            "r_squared": np.nan,
            "converged": False,
            "iterations": 0,
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "covariance_method": "Poisson QMLE score/Hessian city-clustered CR1 sandwich",
            "covariance_meta": {},
            "residuals": np.full(n, np.nan),
            "fitted": np.full(n, np.nan),
            "offset": "lnW",
            "loglik": np.nan,
            "deviance": np.nan,
            "pearson_chi2": np.nan,
        }


def coefficient_table(fits: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        fit = fits[model_id]
        for idx, term in enumerate(fit["names"]):
            rows.append(
                {
                    "model_id": model_id,
                    "method": fit["method"],
                    "outcome": fit["outcome"],
                    "term": term,
                    "term_type": fit["term_types"].get(term, "other"),
                    "coefficient": float(fit["beta"][idx]) if np.isfinite(fit["beta"][idx]) else np.nan,
                    "se": float(fit["se"][idx]) if np.isfinite(fit["se"][idx]) else np.nan,
                    "ci_low": float(fit["ci_low"][idx]) if np.isfinite(fit["ci_low"][idx]) else np.nan,
                    "ci_high": float(fit["ci_high"][idx]) if np.isfinite(fit["ci_high"][idx]) else np.nan,
                    "p_value": float(fit["p_value"][idx]) if np.isfinite(fit["p_value"][idx]) else np.nan,
                    "n": fit["n"],
                    "cities": fit["G"],
                    "rank": fit["rank"],
                    "dim": fit["dim"],
                    "df": fit["df"],
                    "covariance_method": fit["covariance_method"],
                    "offset": fit["offset"],
                    "fit_status": fit["status"],
                    "failure_reason": fit["failure_reason"],
                }
            )
    return pd.DataFrame(rows)


def covariance_table(fits: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        fit = fits[model_id]
        covariance = fit["covariance"]
        if not np.isfinite(covariance).any():
            continue
        names = fit["names"]
        for i, term_i in enumerate(names):
            for j, term_j in enumerate(names):
                rows.append({"model_id": model_id, "term_i": term_i, "term_j": term_j, "covariance": float(covariance[i, j])})
    return pd.DataFrame(rows)


def joint_test(fit: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    names = list(fit["names"])
    indices = {term: i for i, term in enumerate(names)}
    theta_names = ["H1", "H2"]
    if not all(term in indices for term in theta_names) or fit["status"] not in {"estimated", "not_converged"}:
        return {
            "model_id": model_id,
            "test": "theta1=theta2=0",
            "restriction": "H1 and H2 coefficients jointly zero",
            "statistic": np.nan,
            "statistic_type": "Wald_chi2",
            "F_statistic": np.nan,
            "df_num": 2,
            "df_denom": fit.get("df", np.nan),
            "p_value_wald_chi2": np.nan,
            "p_value_F": np.nan,
            "n": fit.get("n", np.nan),
            "cities": fit.get("G", np.nan),
            "rank": fit.get("rank", np.nan),
            "covariance_method": fit.get("covariance_method", ""),
            "status": "not_estimable",
            "failure_reason": fit.get("failure_reason", "nonlinear terms unavailable"),
        }
    idx = [indices[term] for term in theta_names]
    theta = fit["beta"][idx]
    covariance = fit["covariance"][np.ix_(idx, idx)]
    try:
        inv = np.linalg.inv(covariance)
        wald = float(theta.T @ inv @ theta)
        f_stat = wald / 2.0
        p_wald = float(stats.chi2.sf(wald, 2))
        p_f = float(stats.f.sf(f_stat, 2, fit["df"]))
        status = "estimated"
        reason = ""
    except Exception as exc:
        wald = f_stat = p_wald = p_f = np.nan
        status = "failed"
        reason = f"{type(exc).__name__}: {exc}"
    return {
        "model_id": model_id,
        "test": "theta1=theta2=0",
        "restriction": "H1 and H2 coefficients jointly zero",
        "statistic": wald,
        "statistic_type": "Wald_chi2",
        "F_statistic": f_stat,
        "df_num": 2,
        "df_denom": fit["df"],
        "p_value_wald_chi2": p_wald,
        "p_value_F": p_f,
        "n": fit["n"],
        "cities": fit["G"],
        "rank": fit["rank"],
        "covariance_method": fit["covariance_method"],
        "status": status,
        "failure_reason": reason,
    }


def model_comparison(fits: Mapping[str, Mapping[str, Any]], joint: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        fit = fits[model_id]
        index = {term: i for i, term in enumerate(fit["names"])}
        i_x = index.get("x")
        slope = fit["beta"][i_x] if i_x is not None else np.nan
        slope_se = fit["se"][i_x] if i_x is not None else np.nan
        row = {
            "model_id": model_id,
            "method": fit["method"],
            "outcome": fit["outcome"],
            "n": fit["n"],
            "cities": fit["G"],
            "rank": fit["rank"],
            "dimension": fit["dim"],
            "df_city_cluster": fit["df"],
            "slope_x": slope,
            "slope_x_se": slope_se,
            "slope_x_ci_low": fit["ci_low"][i_x] if i_x is not None else np.nan,
            "slope_x_ci_high": fit["ci_high"][i_x] if i_x is not None else np.nan,
            "rss": fit.get("rss", np.nan),
            "r_squared": fit.get("r_squared", np.nan),
            "loglik": fit.get("loglik", np.nan),
            "deviance": fit.get("deviance", np.nan),
            "pearson_chi2": fit.get("pearson_chi2", np.nan),
            "offset": fit.get("offset", ""),
            "nonlinear_wald_chi2": joint.get("statistic", np.nan) if model_id == "L1" else np.nan,
            "nonlinear_F": joint.get("F_statistic", np.nan) if model_id == "L1" else np.nan,
            "nonlinear_F_p_value": joint.get("p_value_F", np.nan) if model_id == "L1" else np.nan,
            "fit_status": fit["status"],
            "converged": fit["converged"],
            "iterations": fit["iterations"],
            "covariance_method": fit["covariance_method"],
            "failure_reason": fit["failure_reason"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def linear_combination_interval(gradient: np.ndarray, fit: Mapping[str, Any]) -> tuple[float, float, float]:
    if fit["status"] not in {"estimated", "not_converged"} or not np.isfinite(fit["beta"]).all() or not np.isfinite(fit["covariance"]).all():
        return np.nan, np.nan, np.nan
    estimate = float(np.dot(gradient, fit["beta"]))
    variance = float(gradient @ fit["covariance"] @ gradient)
    if not np.isfinite(variance):
        return estimate, np.nan, np.nan
    se = math.sqrt(max(variance, 0.0))
    critical = float(stats.t.ppf(0.975, fit["df"]))
    return estimate, estimate - critical * se, estimate + critical * se


def curve_table(model: pd.DataFrame, fits: Mapping[str, Mapping[str, Any]], knots: Sequence[float], controls: Sequence[str]) -> pd.DataFrame:
    x_obs = model["x"].to_numpy(float)
    x_median = float(np.median(x_obs))
    x_grid = np.linspace(float(np.min(x_obs)), float(np.max(x_obs)), GRID_SIZE)
    h_grid = rcs_basis(x_grid, knots)
    h_ref = rcs_basis(np.array([x_median]), knots)[0]
    dh_grid = rcs_derivative(x_grid, knots)
    rows: list[dict[str, Any]] = []
    counts = {
        "below_p5": int(np.sum(x_obs < knots[0])),
        "p5_p95": int(np.sum((x_obs >= knots[0]) & (x_obs <= knots[3]))),
        "above_p95": int(np.sum(x_obs > knots[3])),
    }
    cities_by_region = {
        "below_p5": int(model.loc[model["x"] < knots[0], "city_key"].nunique()),
        "p5_p95": int(model.loc[model["x"].between(knots[0], knots[3], inclusive="both"), "city_key"].nunique()),
        "above_p95": int(model.loc[model["x"] > knots[3], "city_key"].nunique()),
    }
    for model_id in MODEL_IDS:
        fit = fits[model_id]
        indices = {term: i for i, term in enumerate(fit["names"])}
        for grid_i, x_value in enumerate(x_grid):
            if x_value < knots[0]:
                region = "below_p5"
            elif x_value <= knots[3]:
                region = "p5_p95"
            else:
                region = "above_p95"
            if model_id == "L0":
                gradient_curve = np.zeros(len(fit["names"]), dtype=float)
                gradient_curve[indices["x"]] = x_value - x_median
                gradient_eta = np.zeros(len(fit["names"]), dtype=float)
                gradient_eta[indices["x"]] = 1.0
            else:
                gradient_curve = np.zeros(len(fit["names"]), dtype=float)
                gradient_curve[indices["x"]] = x_value - x_median
                gradient_curve[indices["H1"]] = h_grid[grid_i, 0] - h_ref[0]
                gradient_curve[indices["H2"]] = h_grid[grid_i, 1] - h_ref[1]
                gradient_eta = np.zeros(len(fit["names"]), dtype=float)
                gradient_eta[indices["x"]] = 1.0
                gradient_eta[indices["H1"]] = dh_grid[grid_i, 0]
                gradient_eta[indices["H2"]] = dh_grid[grid_i, 1]
            curve, curve_low, curve_high = linear_combination_interval(gradient_curve, fit)
            eta, eta_low, eta_high = linear_combination_interval(gradient_eta, fit)
            rows.append(
                {
                    "model": model_id,
                    "model_scale": "E[ln A|X]" if model_id in {"L0", "L1"} else "ln E[A|X]",
                    "outcome": fit["outcome"],
                    "x": float(x_value),
                    "x_grid_index": int(grid_i),
                    "curve": curve,
                    "curve_ci_low": curve_low,
                    "curve_ci_high": curve_high,
                    "eta": eta,
                    "eta_ci_low": eta_low,
                    "eta_ci_high": eta_high,
                    "x_median_center": x_median,
                    "center_definition": "f(x)-f(sample median x)",
                    "support_region": region,
                    "support_n_rows_region": counts[region],
                    "support_n_cities_region": cities_by_region[region],
                    "support_n_rows_total": len(model),
                    "support_n_cities_total": model["city_key"].nunique(),
                    "p5": knots[0],
                    "p35": knots[1],
                    "p65": knots[2],
                    "p95": knots[3],
                    "eta_minus_one": eta - 1.0 if np.isfinite(eta) else np.nan,
                    "interval_type": "pointwise city-cluster t(G-1)",
                    "simultaneous_band": False,
                    "basis_formula": basis_formula(),
                    "derivative_formula": derivative_formula(),
                    "fit_status": fit["status"],
                    "failure_reason": fit["failure_reason"],
                }
            )
    return pd.DataFrame(rows)


def write_model_sample(frame: pd.DataFrame) -> None:
    columns = [
        "analysis_row_id",
        "city_key",
        "city",
        "province",
        "year",
        "Q",
        "E",
        "W",
        "A",
        "N",
        "I",
        "lnA",
        "lnN",
        "lnI",
        "x",
        *PRIMARY_WEATHER,
        *FALLBACK_WEATHER,
        "common_controls_available",
        "city_positive_years",
        "singleton_city",
        "candidate_member",
        "model_member",
        "sample_role",
        "exclusion_reason",
    ]
    # Keep the active control columns plus any fallback columns that exist.  A
    # stable table is useful to downstream figures even when fallback is used.
    keep: list[str] = []
    for c in columns:
        if c in frame.columns and c not in keep:
            keep.append(c)
    frame[keep].to_csv(OUTPUT / "nonlinear_model_sample_v12.csv", index=False, encoding="utf-8-sig", float_format="%.15g")


def fits_cache_valid(metadata_path: Path, input_signature: str) -> bool:
    required = [
        OUTPUT / "nonlinear_model_comparison_v12.csv",
        OUTPUT / "nonlinear_coefficients_v12.csv",
        OUTPUT / "nonlinear_joint_tests_v12.csv",
        OUTPUT / "service_elasticity_curve_v12.csv",
        OUTPUT / "nonlinear_model_sample_v12.csv",
        OUTPUT / "nonlinear_covariance_v12.csv",
    ]
    if not metadata_path.exists() or not all(p.exists() for p in required):
        return False
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return meta.get("input_signature") == input_signature and meta.get("script_version") == SCRIPT_VERSION


def write_history(records: list[dict[str, Any]], input_signature: str, fit_mode: str, total_fit_calls_recorded: int | None = None) -> None:
    current_fit_calls = int(sum(r.get("event") == "fit" and r.get("performed", False) for r in records))
    if total_fit_calls_recorded is None:
        total_fit_calls_recorded = current_fit_calls
    payload = {
        "script_version": SCRIPT_VERSION,
        "generated_utc": utc_now(),
        "input_signature": input_signature,
        "fit_mode": fit_mode,
        "records": records,
        "execution_counts": {
            "main_model_fits_this_run": current_fit_calls,
            "main_model_fits_planned": 3,
            "main_model_fits_total_recorded": int(total_fit_calls_recorded),
            "models_planned": list(MODEL_IDS),
        },
    }
    write_json(OUTPUT / "nonlinear_execution_history_v12.json", payload)


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    start = utc_now()
    records: list[dict[str, Any]] = [{"event": "start", "utc": start}]
    candidate, model, info = build_annual_sample()
    knots = np.asarray(info["knots"], dtype=float)
    sample_sig = sample_signature(model, info["controls"], knots)
    specs_path, spec_sig = freeze_specs(info, sample_sig)
    write_model_sample(candidate)
    records.append(
        {
            "event": "freeze",
            "utc": utc_now(),
            "spec_path": rel(specs_path),
            "spec_signature": spec_sig,
            "sample_signature": sample_sig,
            "candidate_rows": info["candidate_rows"],
            "candidate_cities": info["candidate_cities"],
            "model_rows": info["model_rows"],
            "model_cities": info["model_cities"],
            "knots": info["knots"],
            "controls": info["controls"],
            "performed": True,
        }
    )
    inputs = {"annual_service_metrics_v9": sha256(ANNUAL_INPUT), "weather_city_year_v4": sha256(WEATHER_INPUT), "script": sha256(HERE)}
    input_signature = hashlib.sha256(
        json.dumps({"inputs": inputs, "spec_signature": spec_sig, "sample_signature": sample_sig, "script_version": SCRIPT_VERSION}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    metadata_path = OUTPUT / "nonlinear_metadata_v12.json"
    cached = fits_cache_valid(metadata_path, input_signature)
    if cached:
        fit_mode = "cache_reuse"
        fits: dict[str, Any] = {}
        # The cache path is deliberately post-fit only.  The existing tables
        # are sufficient for the runner and avoid any accidental refit.
        for model_id in MODEL_IDS:
            fits[model_id] = {"model_id": model_id, "status": "cache_reuse"}
        records.extend({"event": "fit", "model_id": model_id, "performed": False, "status": "cache_reuse", "utc": utc_now()} for model_id in MODEL_IDS)
        existing_meta = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        old_history = existing_meta.get("fit_history", {}) if isinstance(existing_meta.get("fit_history", {}), dict) else {}
        total_fit_calls_recorded = int(old_history.get("total_fit_calls_recorded", existing_meta.get("execution_counts", {}).get("main_model_fits_total_recorded", existing_meta.get("execution_counts", {}).get("main_model_fits_this_run", 0))))
        existing_meta["last_run_utc"] = utc_now()
        existing_meta["fit_mode"] = fit_mode
        existing_meta["execution_counts"] = {"main_model_fits_this_run": 0, "main_model_fits_planned": 3, "main_model_fits_total_recorded": total_fit_calls_recorded, "cached": True}
        existing_meta["fit_history"] = {"first_fresh_fit_count": int(old_history.get("first_fresh_fit_count", total_fit_calls_recorded)), "total_fit_calls_recorded": total_fit_calls_recorded, "last_fresh_fit_utc": old_history.get("last_fresh_fit_utc", existing_meta.get("generated_utc", ""))}
        write_json(metadata_path, existing_meta)
        write_history(records, input_signature, fit_mode, total_fit_calls_recorded)
        return existing_meta

    fit_mode = "fresh_fit"
    fits = {}
    fit_records: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        X, names, meta = make_design(model, model_id, knots, info["controls"])
        meta = {**meta, "model_id": model_id}
        if model_id == "P1":
            fit = fit_ppml(model, X, names, meta)
        else:
            fit = fit_ols(model, X, names, meta)
        fits[model_id] = fit
        fit_records.append(
            {
                "event": "fit",
                "model_id": model_id,
                "performed": True,
                "status": fit["status"],
                "converged": fit["converged"],
                "iterations": fit["iterations"],
                "n": fit["n"],
                "cities": fit["G"],
                "rank": fit["rank"],
                "dimension": fit["dim"],
                "failure_reason": fit["failure_reason"],
                "utc": utc_now(),
            }
        )
    records.extend(fit_records)

    tests = [joint_test(fits["L1"], "L1")]
    comparison = model_comparison(fits, tests[0])
    coefficients = coefficient_table(fits)
    covariance = covariance_table(fits)
    curve = curve_table(model, fits, knots, info["controls"])
    comparison.to_csv(OUTPUT / "nonlinear_model_comparison_v12.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    coefficients.to_csv(OUTPUT / "nonlinear_coefficients_v12.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    pd.DataFrame(tests).to_csv(OUTPUT / "nonlinear_joint_tests_v12.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    curve.to_csv(OUTPUT / "service_elasticity_curve_v12.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    covariance.to_csv(OUTPUT / "nonlinear_covariance_v12.csv", index=False, encoding="utf-8-sig", float_format="%.15g")

    meta_models = {}
    for model_id in MODEL_IDS:
        fit = fits[model_id]
        meta_models[model_id] = {
            "method": fit["method"],
            "outcome": fit["outcome"],
            "status": fit["status"],
            "converged": fit["converged"],
            "iterations": fit["iterations"],
            "n": fit["n"],
            "cities": fit["G"],
            "rank": fit["rank"],
            "dimension": fit["dim"],
            "df": fit["df"],
            "failure_reason": fit["failure_reason"],
            "covariance_method": fit["covariance_method"],
            "covariance_meta": fit["covariance_meta"],
            "offset": fit["offset"],
            "iterations": fit["iterations"],
        }
    metadata = {
        "script": rel(HERE),
        "script_version": SCRIPT_VERSION,
        "generated_utc": utc_now(),
        "last_run_utc": utc_now(),
        "fit_mode": fit_mode,
        "input_signature": input_signature,
        "sample_signature": sample_sig,
        "spec_signature": spec_sig,
        "input_hashes": inputs,
        "inputs": {"annual_service_metrics_v9": rel(ANNUAL_INPUT), "weather_city_year_v4": rel(WEATHER_INPUT)},
        "sample": info,
        "models": meta_models,
        "execution_counts": {"main_model_fits_this_run": 3, "main_model_fits_planned": 3, "main_model_fits_total_recorded": 3, "cached": False},
        "fit_history": {"first_fresh_fit_count": 3, "total_fit_calls_recorded": 3, "last_fresh_fit_utc": utc_now()},
        "fit_calls": {"L0": 1, "L1": 1, "P1": 1},
        "rcs": {"knots": info["knots"], "basis_formula": basis_formula(), "derivative_formula": derivative_formula(), "quantile_method": info["quantile_method"], "interval_type": "pointwise only; no simultaneous band claimed"},
        "inference": {
            "ols": "city cluster CR1 covariance, factor (G/(G-1))*((n-1)/(n-rank)), t(G-1)",
            "ppml": "Poisson score/Hessian sandwich with city-cluster sums; factor (G/(G-1))*((n-1)/(n-rank)), t(G-1) for pointwise intervals",
            "ppml_offset": "ln(W); W is exposure only and not frequency weight",
        },
        "actual_adjustments": [
            "The annual panel source contains exactly the frozen 464 positive Q/E/W rows; no sample substitution was needed.",
            "Nineteen singleton positive-year cities were excluded once before fitting because city FE provide no within-city slope information.",
            "The V4 five-variable HDD18/CDD24/weather set was complete for all 464 candidate rows; no four-weather fallback was used.",
            "The model-sample export was de-duplicated after writing active and fallback weather labels; this changed table formatting only and did not change inputs, rows, knots, or estimates.",
        ],
        "official_method_sources": [
            "https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.html",
            "https://scorreia.com/help/ppmlhdfe.html",
            "https://doi.org/10.1162/rest.88.4.641",
        ],
        "outputs": {
            p.name: {"path": rel(p), "bytes": int(p.stat().st_size), "sha256": sha256(p)}
            for p in OUTPUT.glob("nonlinear_*.csv")
            if p.is_file()
        },
    }
    write_json(metadata_path, metadata)
    write_history(records, input_signature, fit_mode, 3)
    summary = {
        "fit_mode": fit_mode,
        "candidate_rows": info["candidate_rows"],
        "candidate_cities": info["candidate_cities"],
        "model_rows": info["model_rows"],
        "model_cities": info["model_cities"],
        "knots": info["knots"],
        "statuses": {m: fits[m]["status"] for m in MODEL_IDS},
        "L1_nonlinear_F": tests[0]["F_statistic"],
        "L1_nonlinear_F_p": tests[0]["p_value_F"],
    }
    print(json.dumps(json_safe(summary), ensure_ascii=False))
    return metadata


if __name__ == "__main__":
    main()

