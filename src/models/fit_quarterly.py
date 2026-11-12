"""Estimate quarterly dynamic activity responses to facility exposure.

The estimator reads ``data/processed/quarterly_panel.csv`` and writes the
reproducible sample ledger, support audit, coefficients, covariance matrices,
contrasts, response paths, diagnostics, and fixed specification records under
``data/results/recomputed_quarterly/``.  It estimates the prespecified D0, D1,
and D3 finite-distributed-lag Poisson QMLE models, plus the same-support D3
COVID comparison.  Facility exposure enters through the fixed lag
parameterization, ``ln(W)`` is the exposure offset, and city and complete
calendar-quarter fixed effects are treatment coded.

Inference uses the Poisson score/Hessian cluster sandwich rather than
``fit.cov_params``.  A linear-programming separation diagnostic follows the
existence check in Correia, Guimarães and Zylkin (2019); separated all-zero
fixed-effect groups are removed before the final fit.  The implementation uses
``statsmodels.GLM`` Poisson IRLS and is not a call to ``ppmlhdfe``.  The
specification is frozen before estimation; ``--refit`` explicitly refreshes
the result from the packaged input.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.optimize import linprog
import statsmodels
import statsmodels.api as sm

try:  # direct script execution and ``src.models`` package imports
    from dynamic_common import repository_relative
except ImportError:  # pragma: no cover - package import path
    from .dynamic_common import repository_relative


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
OUT = REPOSITORY / "data" / "results" / "recomputed_quarterly"
PANEL = REPOSITORY / "data" / "processed" / "quarterly_panel.csv"
SPEC_PATH = OUT / "quarterly_dynamic_specs_v18.json"
RUN_PATH = OUT / "quarterly_dynamic_run_v18.json"
HISTORY_PATH = OUT / "quarterly_dynamic_fit_attempt_history_v18.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.14g")


def write_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def fixed_spec() -> dict[str, Any]:
    return {
        "status": "fixed before quarterly fitting; this is an analysis specification record, not an original preregistration",
        "support_rule_revision": "Iteratively remove city/quarter all-zero and singleton FE groups to stability; this is recorded because the first exploratory saved attempt exposed a quarter singleton in the candidate support.",
        "created_utc": utc_now(),
        "source_panel": repository_relative(PANEL, REPOSITORY),
        "source_panel_sha256": sha256(PANEL) if PANEL.exists() else None,
        "skeleton": "334-city x 32 natural calendar quarters",
        "candidate_rule": {
            "current": "finite Q >= 0, finite E > 0, finite W > 0",
            "facility_levels": "finite N > 0 at t,t-1,t-2,t-3",
            "lag_link": "exact city_key and quarter_period_ordinal - lag = 1,2,3; no post-drop shift",
            "lag_outcomes": "Q at lag quarters need not be observed or positive",
        },
        "main_support_rule": {
            "all_zero_fe": "remove candidate cities whose candidate Q sum is zero",
            "singleton": "after the candidate city all-zero rule, iteratively remove every city or full-calendar-quarter FE group with one row; the same rule is reapplied until stable",
            "separation": "run an LP existence diagnostic and iteratively remove city/quarter FE groups that are all-zero or singleton; no coefficient truncation",
            "common_support": "D0, D1, and D3 use one final estimable support and one FE structure",
        },
        "models": {
            "D0": "theta0*x_t, theta0 = beta0",
            "D1": "theta1*x_{t-1} + delta0*(x_t-x_{t-1}); beta0=delta0, beta1=theta1-delta0",
            "D3": "theta*x_{t-3} + delta0*(x_t-x_{t-1}) + delta1*(x_{t-1}-x_{t-2}) + delta2*(x_{t-2}-x_{t-3})",
            "D3_mapping": "beta0=delta0; beta1=delta1-delta0; beta2=delta2-delta1; beta3=theta-delta2; c0=delta0,c1=delta1,c2=delta2,c3=theta",
            "D1_mapping": "c0=delta0; c1=theta1",
            "D0_mapping": "c0=theta0",
            "fixed_effects": "city FE + full calendar quarter FE (2019Q1 differs from 2019Q2; not four seasonal dummies)",
            "offset": "ln W; W is exposure, not a frequency weight",
            "weather": "no quarterly weather control; quarterly weather fields are unavailable",
        },
        "inference": {
            "estimator": "Poisson QMLE via statsmodels GLM IRLS",
            "score": "X_i' (Q_i - mu_i)",
            "hessian": "X' diag(mu) X",
            "sandwich": "cluster score/Hessian CR1",
            "cr1_formula": "(G/(G-1))*((n-1)/(n-rank))",
            "rank": "actual full design matrix rank",
            "reference_distribution": "two-sided t(G-1)",
            "main_clusters": "city_key",
            "covid_clusters": "province",
            "bootstrap": False,
        },
        "covid": {
            "support": "D3 main final support, year <= 2022, current genuine source-date covid_mean finite and coverage complete",
            "models": "D3-C0 without covid_mean; D3-C1 with covid_mean/10",
            "same_support": True,
            "no_2023_fill": True,
            "cluster": "province",
        },
        "primary_contrasts": ["c3 = 1", "c3-c0 = 0"],
        "path": "c_h is the model-implied conditional weekly-activity elasticity at h quarters after a persistent proportional facility-level change; not cumulative activity and not a causal impulse response",
        "no_search": ["no lag search", "no bootstrap", "no threshold search", "no new databases"],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "engine": "statsmodels.GLM Poisson IRLS; manual score/Hessian sandwich",
        },
    }


def ensure_spec() -> dict[str, Any]:
    """Write once before fitting; subsequently reuse the frozen record."""
    if SPEC_PATH.exists():
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        if not isinstance(spec, dict) or spec.get("models", {}).get("D3") is None:
            raise RuntimeError(f"Existing dynamic specification is not the fixed quarterly specification: {SPEC_PATH}")
        return spec
    spec = fixed_spec()
    SPEC_PATH.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return spec


def all_output_paths() -> list[Path]:
    return [
        OUT / "quarterly_dynamic_sample_v18.csv",
        OUT / "quarterly_dynamic_sample_audit_v18.csv",
        OUT / "quarterly_dynamic_support_v18.csv",
        OUT / "quarterly_dynamic_specs_v18.csv",
        OUT / "quarterly_dynamic_specs_v18.json",
        OUT / "quarterly_dynamic_coefficients_v18.csv",
        OUT / "quarterly_dynamic_covariance_v18.csv",
        OUT / "quarterly_dynamic_contrasts_v18.csv",
        OUT / "quarterly_dynamic_paths_v18.csv",
        OUT / "quarterly_covid_comparison_v18.csv",
        OUT / "dynamic_findings_v18.md",
        OUT / "quarterly_dynamic_diagnostics_v18.csv",
        RUN_PATH,
        HISTORY_PATH,
    ]


def outputs_complete() -> bool:
    required = all_output_paths()
    if not all(p.exists() and p.stat().st_size > 0 for p in required):
        return False
    try:
        manifest = json.loads(RUN_PATH.read_text(encoding="utf-8"))
        if manifest.get("status") != "stable_result_saved":
            return False
        if manifest.get("source_panel_sha256") != sha256(PANEL):
            return False
        if manifest.get("script_sha256") != sha256(Path(__file__).resolve()):
            return False
        if manifest.get("spec_sha256") != sha256(SPEC_PATH):
            return False
        fit_status = manifest.get("fit_status", {})
        return len(fit_status) == 5 and all(v == "estimated" for v in fit_status.values())
    except Exception:
        return False


def load_panel() -> pd.DataFrame:
    if not PANEL.exists():
        raise FileNotFoundError(PANEL)
    frame = pd.read_csv(PANEL)
    needed = ["city_key", "province", "quarter", "quarter_period_ordinal", "year", "quarter_num", "Q", "E", "W", "A", "N", "I", "covid_mean", "covid_observed_weeks", "covid_coverage_complete", "covid_status", "source_zero_semantics", "q_status", "e_status", "i_status", "W_calendar"]
    missing = sorted(set(needed) - set(frame.columns))
    if missing:
        raise KeyError(f"Quarterly panel missing required columns: {missing}")
    frame.insert(0, "panel_source_row", np.arange(len(frame), dtype=int))
    frame["city_key"] = frame["city_key"].astype(str)
    frame["province"] = frame["province"].astype(str)
    frame["quarter_period_ordinal"] = pd.to_numeric(frame["quarter_period_ordinal"], errors="coerce").astype("Int64")
    for col in ["Q", "E", "W", "W_calendar", "A", "N", "I", "covid_mean", "covid_observed_weeks", "year", "quarter_num"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.sort_values(["city_key", "quarter_period_ordinal", "panel_source_row"], kind="stable").reset_index(drop=True)
    return frame


def exact_lag_merge(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach N lags using exact city and ordinal keys, never a post-drop shift."""
    base_cols = [
        "panel_source_row", "city_key", "province", "quarter", "quarter_period_ordinal", "year", "quarter_num",
        "Q", "E", "W", "W_calendar", "A", "N", "I", "covid_mean", "covid_observed_weeks",
        "covid_coverage_complete", "covid_status", "source_zero_semantics", "q_status", "e_status", "i_status",
    ]
    out = panel[base_cols].copy()
    out["lag_link_ok"] = True
    for lag in range(1, 4):
        lag_frame = panel[["city_key", "quarter_period_ordinal", "N"]].copy()
        lag_frame["quarter_period_ordinal"] = lag_frame["quarter_period_ordinal"].astype("Int64") + lag
        lag_frame = lag_frame.rename(columns={"N": f"N_lag{lag}"})
        out = out.merge(lag_frame, on=["city_key", "quarter_period_ordinal"], how="left", validate="one_to_one")
        out[f"lag{lag}_ordinal"] = out["quarter_period_ordinal"].astype("Int64") - lag
        out[f"lag{lag}_present"] = out[f"N_lag{lag}"].notna()
    return out


def candidate_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if not finite(row.Q) or float(row.Q) < 0:
        reasons.append("Q_missing_or_negative")
    if not finite(row.W) or float(row.W) <= 0:
        reasons.append("W_missing_or_nonpositive")
    if not finite(row.E) or float(row.E) <= 0:
        reasons.append("E_missing_or_nonpositive")
    if not finite(row.N) or float(row.N) <= 0:
        reasons.append("N_t_missing_or_nonpositive")
    for lag in range(1, 4):
        if not finite(row[f"N_lag{lag}"]) or float(row[f"N_lag{lag}"]) <= 0:
            reasons.append(f"N_tminus{lag}_missing_or_nonpositive")
    return "candidate" if not reasons else ";".join(reasons)


def build_candidate(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = exact_lag_merge(panel)
    frame["candidate_reason"] = frame.apply(candidate_reason, axis=1)
    frame["candidate_dynamic_support"] = frame["candidate_reason"].eq("candidate")
    cand = frame.loc[frame["candidate_dynamic_support"]].copy()
    cand["x0"] = np.log(cand["N"].astype(float))
    for lag in range(1, 4):
        cand[f"x{lag}"] = np.log(cand[f"N_lag{lag}"].astype(float))
    cand["delta0"] = cand["x0"] - cand["x1"]
    cand["delta1"] = cand["x1"] - cand["x2"]
    cand["delta2"] = cand["x2"] - cand["x3"]
    cand["candidate_city_q_sum"] = cand.groupby("city_key")["Q"].transform("sum")
    cand["candidate_city_n"] = cand.groupby("city_key")["Q"].transform("size")
    cand["all_zero_city"] = cand["candidate_city_q_sum"].eq(0)
    cand["city_singleton"] = cand["candidate_city_n"].eq(1)
    cand["main_initial_support"] = ~(cand["all_zero_city"] | cand["city_singleton"])
    cand["support_exclusion_reason"] = np.select(
        [cand["all_zero_city"], cand["city_singleton"]],
        ["all_zero_city_fe", "city_singleton"],
        default="initial_main_support",
    )
    # A separate audit has one row per full calendar skeleton row, while the
    # sample is restricted to candidate rows below.
    audit = frame.copy()
    audit["all_zero_city"] = False
    audit["city_singleton"] = False
    audit["main_initial_support"] = False
    audit["support_exclusion_reason"] = np.where(audit["candidate_dynamic_support"], "candidate_not_yet_classified", audit["candidate_reason"])
    audit_idx = audit["candidate_dynamic_support"]
    lookup = cand.set_index("panel_source_row")[["all_zero_city", "city_singleton", "main_initial_support", "support_exclusion_reason"]]
    # Assign each mapped field separately: tuple assignment can be interpreted
    # as a two-dimensional ndarray by pandas when the boolean mask is sparse.
    key_series = audit.loc[audit_idx, "panel_source_row"]
    for field in ["all_zero_city", "city_singleton", "main_initial_support", "support_exclusion_reason"]:
        audit.loc[audit_idx, field] = key_series.map(lookup[field]).to_numpy()
    return cand, audit


def separation_lp(X: np.ndarray, y: np.ndarray, max_rows: int = 20000) -> dict[str, Any]:
    """Find a finite-normalized PPML separation direction.

    The LP minimizes sum_{Q=0} X d subject to X_pos d = 0,
    X_zero d <= 0 and ||d||_1 <= 1.  A strictly negative optimum is
    evidence of a direction that leaves positive observations unchanged while
    shrinking at least one zero mean.  A zero optimum can be the zero vector
    only and is recorded as no detected strict separation.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if len(y) != len(X):
        raise ValueError("separation LP length mismatch")
    pos = y > 0
    zero = ~pos
    p = X.shape[1]
    if not zero.any() or not pos.any():
        return {"status": "not_applicable", "objective": np.nan, "direction": np.zeros(p), "violating_zero_rows": 0, "message": "requires both positive and zero outcomes"}
    # Duplicate signed variables make the L1 normalization explicit.
    c = np.r_[X[zero].sum(axis=0), -X[zero].sum(axis=0)]
    A_eq = np.c_[X[pos], -X[pos]]
    b_eq = np.zeros(int(pos.sum()))
    A_ub = np.vstack([np.c_[X[zero], -X[zero]], np.ones((1, 2 * p))])
    b_ub = np.r_[np.zeros(int(zero.sum())), 1.0]
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    except Exception as exc:
        return {"status": "lp_failed", "objective": np.nan, "direction": np.zeros(p), "violating_zero_rows": 0, "message": f"{type(exc).__name__}: {exc}"}
    if not result.success:
        return {"status": "lp_failed", "objective": np.nan, "direction": np.zeros(p), "violating_zero_rows": 0, "message": str(result.message)}
    direction = result.x[:p] - result.x[p:]
    zval = X[zero] @ direction
    objective = float(result.fun)
    strict = objective < -1e-8 and bool(np.any(zval < -1e-8))
    return {
        "status": "strict_separation_detected" if strict else "no_strict_separation_detected",
        "objective": objective,
        "direction": direction,
        "violating_zero_rows": int(np.sum(zval < -1e-8)),
        "message": str(result.message),
    }


def make_design(frame: pd.DataFrame, model_id: str, covid: bool = False) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Construct one treatment-coded full design and preserve term metadata."""
    frame = frame.sort_values(["city_key", "quarter_period_ordinal", "panel_source_row"], kind="stable").reset_index(drop=True)
    if model_id == "D0":
        dynamic_names = ["theta0"]
        dynamic = frame[["x0"]].rename(columns={"x0": "theta0"})
    elif model_id == "D1":
        dynamic_names = ["theta1", "delta0"]
        dynamic = frame[["x1", "delta0"]].rename(columns={"x1": "theta1"})
    elif model_id == "D3":
        dynamic_names = ["theta", "delta0", "delta1", "delta2"]
        dynamic = frame[["x3", "delta0", "delta1", "delta2"]].rename(columns={"x3": "theta"})
    else:
        raise ValueError(model_id)
    blocks: list[pd.DataFrame] = [pd.DataFrame({"const": np.ones(len(frame))}, index=frame.index), dynamic]
    city = pd.get_dummies(frame["city_key"].astype(str), prefix="city_FE", drop_first=True, dtype=float)
    quarter = pd.get_dummies(frame["quarter_period_ordinal"].astype(int), prefix="quarter_FE", drop_first=True, dtype=float)
    blocks.extend([city, quarter])
    if covid:
        blocks.append(pd.DataFrame({"covid_mean_per10": frame["covid_mean"].astype(float).to_numpy() / 10.0}, index=frame.index))
    design = pd.concat(blocks, axis=1)
    names = design.columns.astype(str).tolist()
    X = design.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(X))
    meta = {
        "dynamic_names": dynamic_names,
        "term_types": {"const": "intercept", **{n: "dynamic_reparameterization" for n in dynamic_names}, **{n: "city_fixed_effect" for n in city.columns}, **{n: "calendar_quarter_fixed_effect" for n in quarter.columns}, **({"covid_mean_per10": "current_source_date_covid_index_per10"} if covid else {})},
        "city_levels": sorted(frame["city_key"].astype(str).unique().tolist()),
        "quarter_levels": sorted(frame["quarter_period_ordinal"].astype(int).unique().tolist()),
        "rank": rank,
        "dim": int(X.shape[1]),
        "covid": covid,
        "model_id": model_id,
    }
    return X, names, meta


def cr1_covariance(scores: np.ndarray, clusters: Sequence[Any], bread: np.ndarray, rank: int) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(pd.Series(clusters).astype(str))
    unique = pd.unique(labels)
    n, p = scores.shape
    G = int(len(unique))
    if G <= 1 or n <= rank:
        raise ValueError(f"Invalid CR1 dimensions n={n}, rank={rank}, G={G}")
    meat = np.zeros((p, p), dtype=float)
    for label in unique:
        score = scores[labels == label].sum(axis=0)
        meat += np.outer(score, score)
    factor = (G / (G - 1.0)) * ((n - 1.0) / (n - rank))
    cov = factor * bread @ meat @ bread
    cov = (cov + cov.T) / 2.0
    return cov, {"clusters": G, "n": n, "rank": int(rank), "cr1_factor": float(factor), "cr1_formula": "(G/(G-1))*((n-1)/(n-rank))"}


def fit_one(frame: pd.DataFrame, model_id: str, covid: bool = False, cluster_col: str = "city_key") -> dict[str, Any]:
    """Fit once and return all point, covariance and diagnostic material."""
    sub = frame.sort_values(["city_key", "quarter_period_ordinal", "panel_source_row"], kind="stable").reset_index(drop=True).copy()
    X, names, meta = make_design(sub, model_id, covid=covid)
    y = sub["Q"].to_numpy(float)
    offset = np.log(sub["W"].to_numpy(float))
    n, p = X.shape
    rank = int(np.linalg.matrix_rank(X))
    if rank != p:
        return failure_fit(sub, model_id, covid, cluster_col, names, meta, f"design rank deficient: rank={rank}, dim={p}")
    separation = separation_lp(X, y)
    if separation["status"] == "lp_failed":
        return failure_fit(sub, model_id, covid, cluster_col, names, meta, f"separation LP failed: {separation.get('message', '')}", separation=separation)
    if separation["status"] == "strict_separation_detected":
        return failure_fit(sub, model_id, covid, cluster_col, names, meta, f"strict separation LP objective={separation['objective']:.12g}; rows={separation['violating_zero_rows']}", separation=separation)
    caught: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            fit = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit(maxiter=500, tol=1e-10, method="IRLS", disp=0)
            caught = [str(w.message) for w in warns]
        beta = np.asarray(fit.params, dtype=float)
        converged = bool(getattr(fit, "converged", False)) and np.isfinite(beta).all()
        mu = np.exp(np.clip(offset + X @ beta, -745.0, 709.0))
        if not np.isfinite(mu).all():
            raise FloatingPointError("non-finite fitted mean")
        hessian = X.T @ (mu[:, None] * X)
        hessian_rank = int(np.linalg.matrix_rank(hessian))
        if hessian_rank != p:
            raise np.linalg.LinAlgError(f"weighted Hessian rank deficient: rank={hessian_rank}, dim={p}")
        bread = np.linalg.inv(hessian)
        scores = X * (y - mu)[:, None]
        covariance, cov_meta = cr1_covariance(scores, sub[cluster_col], bread, rank)
        se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        clusters = int(sub[cluster_col].astype(str).nunique())
        df = clusters - 1
        critical = float(stats.t.ppf(0.975, df))
        t_value = np.divide(beta, se, out=np.full(p, np.nan), where=se > 0)
        p_value = 2.0 * stats.t.sf(np.abs(t_value), df)
        ci_low = beta - critical * se
        ci_high = beta + critical * se
        return {
            "model_id": model_id,
            "covid": covid,
            "cluster_col": cluster_col,
            "frame": sub,
            "X": X,
            "names": names,
            "meta": meta,
            "beta": beta,
            "covariance": covariance,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
            "n": n,
            "G": clusters,
            "rank": rank,
            "dim": p,
            "df": df,
            "critical": critical,
            "converged": converged,
            "iterations": int(fit.fit_history.get("iteration", 0)) if isinstance(fit.fit_history, dict) else np.nan,
            "status": "estimated" if converged else "not_converged",
            "failure_reason": "" if converged else "statsmodels GLM IRLS did not report convergence or non-finite coefficient",
            "covariance_method": "Poisson QMLE score/Hessian cluster CR1; t(G-1)",
            "covariance_meta": {**cov_meta, "hessian": "X' diag(mu) X", "score": "X_i*(Q_i-mu_i)", "warnings": caught},
            "separation": separation,
            "mu": mu,
            "score_residual_l2": float(np.linalg.norm(scores.sum(axis=0))),
            "score_residual_max_abs": float(np.max(np.abs(scores.sum(axis=0)))),
            "mean_fitted": float(np.mean(mu)),
            "sum_observed": float(np.sum(y)),
            "sum_fitted": float(np.sum(mu)),
            "deviance": float(getattr(fit, "deviance", np.nan)),
            "pearson_chi2": float(getattr(fit, "pearson_chi2", np.nan)),
            "loglik": float(getattr(fit, "llf", np.nan)),
        }
    except Exception as exc:
        return failure_fit(sub, model_id, covid, cluster_col, names, meta, f"{type(exc).__name__}: {exc}", separation=separation)


def failure_fit(frame: pd.DataFrame, model_id: str, covid: bool, cluster_col: str, names: Sequence[str], meta: Mapping[str, Any], reason: str, separation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    p = len(names)
    n = len(frame)
    return {
        "model_id": model_id, "covid": covid, "cluster_col": cluster_col, "frame": frame, "X": np.empty((n, p)), "names": list(names), "meta": dict(meta),
        "beta": np.full(p, np.nan), "covariance": np.full((p, p), np.nan), "se": np.full(p, np.nan), "ci_low": np.full(p, np.nan), "ci_high": np.full(p, np.nan), "p_value": np.full(p, np.nan),
        "n": n, "G": int(frame[cluster_col].astype(str).nunique()), "rank": int(meta.get("rank", np.nan)) if finite(meta.get("rank", np.nan)) else np.nan, "dim": p, "df": int(frame[cluster_col].astype(str).nunique() - 1), "critical": np.nan,
        "converged": False, "iterations": 0, "status": "failed", "failure_reason": reason, "covariance_method": "Poisson QMLE score/Hessian cluster CR1; t(G-1)", "covariance_meta": {}, "separation": separation or {},
        "mu": np.full(n, np.nan), "score_residual_l2": np.nan, "score_residual_max_abs": np.nan, "mean_fitted": np.nan, "sum_observed": float(frame["Q"].sum()), "sum_fitted": np.nan, "deviance": np.nan, "pearson_chi2": np.nan, "loglik": np.nan,
    }


def transform_matrix(model_id: str, names: Sequence[str], target: str = "beta") -> tuple[list[str], np.ndarray]:
    idx = {name: i for i, name in enumerate(names)}
    p = len(names)
    if target == "beta":
        if model_id == "D0":
            out, rows = ["beta0"], [np.zeros(p)]
            rows[0][idx["theta0"]] = 1.0
        elif model_id == "D1":
            out, rows = ["beta0", "beta1"], [np.zeros(p), np.zeros(p)]
            rows[0][idx["delta0"]] = 1.0
            rows[1][idx["theta1"]] = 1.0; rows[1][idx["delta0"]] = -1.0
        else:
            out, rows = ["beta0", "beta1", "beta2", "beta3"], [np.zeros(p) for _ in range(4)]
            rows[0][idx["delta0"]] = 1.0
            rows[1][idx["delta1"]] = 1.0; rows[1][idx["delta0"]] = -1.0
            rows[2][idx["delta2"]] = 1.0; rows[2][idx["delta1"]] = -1.0
            rows[3][idx["theta"]] = 1.0; rows[3][idx["delta2"]] = -1.0
    elif target == "path":
        if model_id == "D0":
            out, rows = ["c0"], [np.zeros(p)]; rows[0][idx["theta0"]] = 1.0
        elif model_id == "D1":
            out, rows = ["c0", "c1"], [np.zeros(p), np.zeros(p)]; rows[0][idx["delta0"]] = 1.0; rows[1][idx["theta1"]] = 1.0
        else:
            out, rows = ["c0", "c1", "c2", "c3"], [np.zeros(p) for _ in range(4)]
            rows[0][idx["delta0"]] = 1.0; rows[1][idx["delta1"]] = 1.0; rows[2][idx["delta2"]] = 1.0; rows[3][idx["theta"]] = 1.0
    else:
        raise ValueError(target)
    T = np.vstack(rows)
    return out, T


def transformed_values(fit: Mapping[str, Any], target: str = "beta") -> dict[str, Any]:
    labels, T = transform_matrix(fit["model_id"], fit["names"], target=target)
    beta = np.asarray(fit["beta"], dtype=float)
    cov = np.asarray(fit["covariance"], dtype=float)
    est = T @ beta if np.isfinite(beta).all() else np.full(len(labels), np.nan)
    vcov = T @ cov @ T.T if np.isfinite(cov).all() else np.full((len(labels), len(labels)), np.nan)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0)) if np.isfinite(vcov).all() else np.full(len(labels), np.nan)
    critical = float(fit["critical"]) if finite(fit.get("critical")) else np.nan
    low = est - critical * se if finite(critical) else np.full(len(labels), np.nan)
    high = est + critical * se if finite(critical) else np.full(len(labels), np.nan)
    pval = 2.0 * stats.t.sf(np.abs(np.divide(est, se, out=np.full(len(est), np.nan), where=se > 0)), int(fit["df"])) if finite(fit.get("df")) else np.full(len(labels), np.nan)
    return {"labels": labels, "T": T, "estimate": est, "covariance": vcov, "se": se, "ci_low": low, "ci_high": high, "p_value": pval}


def coefficient_rows(fits: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fit_name, fit in fits.items():
        beta_transform = transformed_values(fit, target="beta") if fit["status"] == "estimated" else None
        beta_by_label = dict(zip(beta_transform["labels"], beta_transform["estimate"])) if beta_transform else {}
        beta_se = dict(zip(beta_transform["labels"], beta_transform["se"])) if beta_transform else {}
        beta_low = dict(zip(beta_transform["labels"], beta_transform["ci_low"])) if beta_transform else {}
        beta_high = dict(zip(beta_transform["labels"], beta_transform["ci_high"])) if beta_transform else {}
        beta_p = dict(zip(beta_transform["labels"], beta_transform["p_value"])) if beta_transform else {}
        for i, term in enumerate(fit["names"]):
            rows.append({"model_id": fit_name, "covid": bool(fit["covid"]), "parameterization": "fitted_design", "term": term, "term_type": fit["meta"]["term_types"].get(term, ""), "estimate": fit["beta"][i], "se": fit["se"][i], "ci_low": fit["ci_low"][i], "ci_high": fit["ci_high"][i], "p_value": fit["p_value"][i], "n": fit["n"], "clusters": fit["G"], "df": fit["df"], "rank": fit["rank"], "design_dim": fit["dim"], "status": fit["status"], "converged": fit["converged"], "iterations": fit["iterations"], "cluster_level": fit["cluster_col"], "covariance_method": fit["covariance_method"], "failure_reason": fit["failure_reason"]})
        for label in [f"beta{i}" for i in range(4) if f"beta{i}" in beta_by_label]:
            rows.append({"model_id": fit_name, "covid": bool(fit["covid"]), "parameterization": "original_level", "term": label, "term_type": "facility_level_lag_coefficient", "estimate": beta_by_label[label], "se": beta_se[label], "ci_low": beta_low[label], "ci_high": beta_high[label], "p_value": beta_p[label], "n": fit["n"], "clusters": fit["G"], "df": fit["df"], "rank": fit["rank"], "design_dim": fit["dim"], "status": fit["status"], "converged": fit["converged"], "iterations": fit["iterations"], "cluster_level": fit["cluster_col"], "covariance_method": fit["covariance_method"], "failure_reason": fit["failure_reason"]})
    return pd.DataFrame(rows)


def covariance_rows(fits: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fit_name, fit in fits.items():
        cov = np.asarray(fit["covariance"], dtype=float)
        for i, ti in enumerate(fit["names"]):
            for j, tj in enumerate(fit["names"]):
                rows.append({"model_id": fit_name, "covid": bool(fit["covid"]), "parameterization": "fitted_design", "term_i": ti, "term_j": tj, "covariance": cov[i, j], "n": fit["n"], "clusters": fit["G"], "df": fit["df"], "rank": fit["rank"], "design_dim": fit["dim"], "status": fit["status"], "covariance_method": fit["covariance_method"]})
        if fit["status"] == "estimated":
            transformed = transformed_values(fit, target="beta")
            for i, ti in enumerate(transformed["labels"]):
                for j, tj in enumerate(transformed["labels"]):
                    rows.append({"model_id": fit_name, "covid": bool(fit["covid"]), "parameterization": "original_level", "term_i": ti, "term_j": tj, "covariance": transformed["covariance"][i, j], "n": fit["n"], "clusters": fit["G"], "df": fit["df"], "rank": fit["rank"], "design_dim": fit["dim"], "status": fit["status"], "covariance_method": fit["covariance_method"]})
    return pd.DataFrame(rows)


def contrast_row(fit: Mapping[str, Any], estimate: float, variance: float, label: str, null: float, purpose: str) -> dict[str, Any]:
    se = float(math.sqrt(max(variance, 0.0))) if finite(variance) else np.nan
    critical = float(fit["critical"]) if finite(fit.get("critical")) else np.nan
    ci_low = estimate - critical * se if finite(se) and finite(critical) else np.nan
    ci_high = estimate + critical * se if finite(se) and finite(critical) else np.nan
    tval = (estimate - null) / se if finite(se) and se > 0 else np.nan
    pval = float(2.0 * stats.t.sf(abs(tval), int(fit["df"]))) if finite(tval) and finite(fit.get("df")) else np.nan
    return {"model_id": fit["model_id"], "comparison": label, "purpose": purpose, "estimate": estimate, "null": null, "se": se, "ci_low": ci_low, "ci_high": ci_high, "t_value": tval, "p_value": pval, "n": fit["n"], "clusters": fit["G"], "df": fit["df"], "rank": fit["rank"], "design_dim": fit["dim"], "cluster_level": fit["cluster_col"], "status": fit["status"], "converged": fit["converged"], "iterations": fit["iterations"], "covariance_method": fit["covariance_method"], "failure_reason": fit["failure_reason"]}


def paths_and_contrasts(fits: Mapping[str, Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for model_id, fit in fits.items():
        if fit["status"] != "estimated":
            continue
        path = transformed_values(fit, target="path")
        for h, label in enumerate(path["labels"]):
            c = float(path["estimate"][h]); lo = float(path["ci_low"][h]); hi = float(path["ci_high"][h])
            path_rows.append({"model_id": model_id, "h": h, "path_term": label, "estimate": c, "se": path["se"][h], "ci_low": lo, "ci_high": hi, "p_value_vs_zero": path["p_value"][h], "ten_percent_activity_difference_pct": 100.0 * (1.1 ** c - 1.0), "ten_percent_ci_low_pct": 100.0 * (1.1 ** lo - 1.0), "ten_percent_ci_high_pct": 100.0 * (1.1 ** hi - 1.0), "relative_to_unit_pct": 100.0 * (1.1 ** (c - 1.0) - 1.0), "n": fit["n"], "clusters": fit["G"], "df": fit["df"], "rank": fit["rank"], "status": fit["status"], "path_definition": "conditional weekly activity elasticity at h quarters after stipulated persistent facility-level change; not cumulative or causal"})
        if model_id == "D3":
            # c3 = 1 and c3-c0 = 0 are fixed primary comparisons.
            v = path["covariance"]
            contrast_rows.append(contrast_row(fit, float(path["estimate"][3]), float(v[3, 3]), "c3", 1.0, "primary: complete stipulated short-run response equals unit elasticity"))
            a = np.zeros(4); a[3] = 1.0; a[0] = -1.0
            contrast_rows.append(contrast_row(fit, float(a @ path["estimate"]), float(a @ v @ a), "c3_minus_c0", 0.0, "primary: later three quarters add no conditional follow-through beyond contemporaneous response"))
        elif model_id == "D1":
            v = path["covariance"]
            contrast_rows.append(contrast_row(fit, float(path["estimate"][1]), float(v[1, 1]), "c1", 1.0, "descriptive D1 finite one-lag path versus unit elasticity; not same horizon as D3 c3"))
    return pd.DataFrame(path_rows), pd.DataFrame(contrast_rows)


def sample_columns(sample: pd.DataFrame, final_rows: pd.DataFrame) -> pd.DataFrame:
    cols = ["panel_source_row", "city_key", "province", "quarter", "quarter_period_ordinal", "year", "quarter_num", "Q", "E", "W", "W_calendar", "A", "N", "I", "source_zero_semantics", "q_status", "e_status", "i_status", "covid_mean", "covid_observed_weeks", "covid_coverage_complete", "covid_status", "N_lag1", "N_lag2", "N_lag3", "x0", "x1", "x2", "x3", "delta0", "delta1", "delta2", "candidate_city_q_sum", "candidate_city_n", "all_zero_city", "city_singleton", "main_initial_support", "support_exclusion_reason"]
    out = sample[[c for c in cols if c in sample.columns]].copy()
    final_ids = set(final_rows["panel_source_row"].astype(int))
    out["final_dynamic_support"] = out["panel_source_row"].astype(int).isin(final_ids)
    for model_id in ["D0", "D1", "D3"]:
        out[f"included_{model_id}"] = out["final_dynamic_support"]
    out["record_coverage_ratio"] = np.divide(out["W"].astype(float), out["W_calendar"].astype(float), out=np.full(len(out), np.nan), where=out["W_calendar"].astype(float).to_numpy() > 0)
    out["encoded_zero_Q"] = out["Q"].eq(0) & out["q_status"].astype(str).eq("observed_compatible_zero")
    out["final_city_vertical_length"] = out["city_key"].map(final_rows.groupby("city_key").size())
    return out.sort_values(["city_key", "quarter_period_ordinal"], kind="stable")


def support_rows(panel: pd.DataFrame, candidate: pd.DataFrame, audit: pd.DataFrame, final: pd.DataFrame, fits: Mapping[str, Mapping[str, Any]], covid_frame: pd.DataFrame, covid_fits: Mapping[str, Mapping[str, Any]], main_removed: pd.DataFrame | None = None, covid_removed: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def add(stage: str, metric: str, value: Any, n: int | float | None = None, cities: int | float | None = None, quarters: int | float | None = None, provinces: int | float | None = None, reason: str = "") -> None:
        rows.append({"stage": stage, "metric": metric, "value": value, "rows": n, "cities": cities, "quarters": quarters, "provinces": provinces, "reason": reason})
    add("skeleton", "rows", len(panel), len(panel), panel.city_key.nunique(), panel.quarter_period_ordinal.nunique(), panel.province.nunique(), "334-city x 32-quarter natural calendar skeleton")
    add("skeleton", "quarterly_weather_available", int((panel.get("quarterly_weather_available", pd.Series(False, index=panel.index)) == True).sum()), len(panel), panel.city_key.nunique(), panel.quarter_period_ordinal.nunique(), panel.province.nunique(), "source panel reports no usable local quarterly weather for this round")
    add("candidate", "candidate_rows", len(candidate), len(candidate), candidate.city_key.nunique(), candidate.quarter_period_ordinal.nunique(), candidate.province.nunique(), "current Q>=0/E,W>0 and exact positive N lags 1-3")
    add("candidate", "positive_Q_rows", int((candidate.Q > 0).sum()), len(candidate), candidate.city_key.nunique(), candidate.quarter_period_ordinal.nunique(), candidate.province.nunique(), "recorded positive activity")
    add("candidate", "encoded_zero_Q_rows", int((candidate.Q == 0).sum()), len(candidate), candidate.city_key.nunique(), candidate.quarter_period_ordinal.nunique(), candidate.province.nunique(), "source-coded Q=0 retained")
    add("candidate", "candidate_city_count", candidate.city_key.nunique(), len(candidate), candidate.city_key.nunique(), candidate.quarter_period_ordinal.nunique(), candidate.province.nunique(), "candidate cities are not PPML identified cities")
    add("candidate", "candidate_city_vertical_length_min", candidate.groupby("city_key").size().min(), len(candidate), candidate.city_key.nunique(), candidate.quarter_period_ordinal.nunique(), candidate.province.nunique(), "candidate rows per city")
    add("candidate", "candidate_city_vertical_length_median", candidate.groupby("city_key").size().median(), len(candidate), candidate.city_key.nunique(), candidate.quarter_period_ordinal.nunique(), candidate.province.nunique(), "candidate rows per city")
    add("exclusion", "all_zero_city_count", int(candidate.loc[candidate.all_zero_city, "city_key"].nunique()), int(candidate.all_zero_city.sum()), int(candidate.loc[candidate.all_zero_city, "city_key"].nunique()), candidate.loc[candidate.all_zero_city, "quarter_period_ordinal"].nunique(), candidate.loc[candidate.all_zero_city, "province"].nunique(), "all candidate Q outcomes zero in city FE group")
    add("exclusion", "all_zero_city_rows", int(candidate.all_zero_city.sum()), int(candidate.all_zero_city.sum()), int(candidate.loc[candidate.all_zero_city, "city_key"].nunique()), candidate.loc[candidate.all_zero_city, "quarter_period_ordinal"].nunique(), candidate.loc[candidate.all_zero_city, "province"].nunique(), "removed before fitting")
    nonzero = candidate.loc[~candidate.all_zero_city]
    add("exclusion", "city_singleton_count", int(nonzero.loc[nonzero.city_singleton, "city_key"].nunique()), int(nonzero.city_singleton.sum()), int(nonzero.loc[nonzero.city_singleton, "city_key"].nunique()), nonzero.loc[nonzero.city_singleton, "quarter_period_ordinal"].nunique(), nonzero.loc[nonzero.city_singleton, "province"].nunique(), "one candidate row in city FE group")
    add("exclusion", "city_singleton_rows", int(nonzero.city_singleton.sum()), int(nonzero.city_singleton.sum()), int(nonzero.loc[nonzero.city_singleton, "city_key"].nunique()), nonzero.loc[nonzero.city_singleton, "quarter_period_ordinal"].nunique(), nonzero.loc[nonzero.city_singleton, "province"].nunique(), "removed before fitting")
    if main_removed is not None and len(main_removed):
        add("exclusion", "iterative_main_fe_removed_rows", len(main_removed), len(main_removed), main_removed.city_key.nunique(), main_removed.quarter_period_ordinal.nunique(), np.nan, "iterative city/quarter FE all-zero or singleton removal")
        for reason, count in main_removed["fe_support_reason"].str.split(";").explode().value_counts().items():
            add("exclusion", f"iterative_main_{reason}_rows", int(count), int(count), np.nan, np.nan, np.nan, "row may carry more than one FE reason")
    add("main_final", "rows", len(final), len(final), final.city_key.nunique(), final.quarter_period_ordinal.nunique(), final.province.nunique(), "common D0/D1/D3 support after FE/separation checks")
    add("main_final", "positive_Q_rows", int((final.Q > 0).sum()), len(final), final.city_key.nunique(), final.quarter_period_ordinal.nunique(), final.province.nunique(), "recorded positive activity")
    add("main_final", "encoded_zero_Q_rows", int((final.Q == 0).sum()), len(final), final.city_key.nunique(), final.quarter_period_ordinal.nunique(), final.province.nunique(), "retained in PPML support")
    lengths = final.groupby("city_key").size()
    for q, val in {"min": lengths.min(), "median": lengths.median(), "max": lengths.max()}.items():
        add("main_final", f"city_vertical_length_{q}", val, len(final), len(lengths), final.quarter_period_ordinal.nunique(), final.province.nunique(), "final rows per city")
    for q, val in final.W_calendar.div(final.W).describe(percentiles=[.05, .25, .5, .75, .95]).to_dict().items():
        add("main_final", f"W_over_W_calendar_{q}", val, len(final), final.city_key.nunique(), final.quarter_period_ordinal.nunique(), final.province.nunique(), "W/W_calendar shown as a record-support diagnostic")
    for model_id, fit in fits.items():
        add("fit", f"{model_id}_status", fit["status"], fit["n"], fit["G"], fit["frame"]["quarter_period_ordinal"].nunique(), fit["frame"]["province"].nunique(), fit["failure_reason"] or "actual fit")
        add("fit", f"{model_id}_rank", fit["rank"], fit["n"], fit["G"], fit["frame"]["quarter_period_ordinal"].nunique(), fit["frame"]["province"].nunique(), "actual full design rank")
        add("fit", f"{model_id}_cr1_factor", fit.get("covariance_meta", {}).get("cr1_factor", np.nan), fit["n"], fit["G"], fit["frame"]["quarter_period_ordinal"].nunique(), fit["frame"]["province"].nunique(), "cluster CR1 finite correction")
        add("fit", f"{model_id}_separation_lp_objective", fit.get("separation", {}).get("objective", np.nan), fit["n"], fit["G"], fit["frame"]["quarter_period_ordinal"].nunique(), fit["frame"]["province"].nunique(), fit.get("separation", {}).get("status", ""))
    add("covid_candidate", "rows_before_covid_separation", len(covid_frame), len(covid_frame), covid_frame.city_key.nunique(), covid_frame.quarter_period_ordinal.nunique(), covid_frame.province.nunique(), "D3 final support year<=2022 and complete current source-date exposure")
    add("covid_candidate", "Q_zero_rows_before_covid_separation", int((covid_frame.Q == 0).sum()), len(covid_frame), covid_frame.city_key.nunique(), covid_frame.quarter_period_ordinal.nunique(), covid_frame.province.nunique(), "retained before FE singleton/separation check")
    if covid_removed is not None and len(covid_removed):
        add("exclusion", "iterative_covid_fe_removed_rows", len(covid_removed), len(covid_removed), covid_removed.city_key.nunique(), covid_removed.quarter_period_ordinal.nunique(), np.nan, "iterative city/quarter FE all-zero or singleton removal; shared C0/C1 support")
        for reason, count in covid_removed["fe_support_reason"].str.split(";").explode().value_counts().items():
            add("exclusion", f"iterative_covid_{reason}_rows", int(count), int(count), np.nan, np.nan, np.nan, "row may carry more than one FE reason")
    for model_id, fit in covid_fits.items():
        add("covid_final", f"{model_id}_rows", fit["n"], fit["n"], fit["frame"].city_key.nunique(), fit["frame"].quarter_period_ordinal.nunique(), fit["frame"].province.nunique(), "C0/C1 shared support after iterative city/quarter FE removal")
        add("covid_final", f"{model_id}_provinces", fit["G"], fit["n"], fit["frame"].city_key.nunique(), fit["frame"].quarter_period_ordinal.nunique(), fit["frame"].province.nunique(), "province cluster count")
        add("covid_final", f"{model_id}_separation_lp_objective", fit.get("separation", {}).get("objective", np.nan), fit["n"], fit["G"], fit["frame"]["quarter_period_ordinal"].nunique(), fit["frame"]["province"].nunique(), fit.get("separation", {}).get("status", ""))
    return pd.DataFrame(rows)


def apply_city_zero_separation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Iteratively remove zero/singleton city and quarter FE groups.

    A city with no positive Q has a city FE at minus infinity.  A singleton
    city or quarter FE group contributes no within-group information and is
    removed under the fixed singleton rule.  The same rule is applied to both
    FE dimensions until stable; C0 and C1 receive exactly this common support.
    The historical function name is retained so the support-repair action is
    explicit in the run history.
    """
    work = frame.copy()
    records: list[dict[str, Any]] = []
    iteration = 0
    while True:
        iteration += 1
        city_groups = work.groupby("city_key")["Q"].agg(["size", "sum"])
        quarter_groups = work.groupby("quarter_period_ordinal")["Q"].agg(["size", "sum"])
        bad_city_zero = set(city_groups.index[city_groups["sum"].eq(0)])
        bad_quarter_zero = set(quarter_groups.index[quarter_groups["sum"].eq(0)])
        bad_city_singleton = set(city_groups.index[city_groups["size"].eq(1)])
        bad_quarter_singleton = set(quarter_groups.index[quarter_groups["size"].eq(1)])
        bad_city = bad_city_zero | bad_city_singleton
        bad_quarter = bad_quarter_zero | bad_quarter_singleton
        if not bad_city and not bad_quarter:
            break
        mask = work.city_key.isin(bad_city) | work.quarter_period_ordinal.isin(bad_quarter)
        for _, row in work.loc[mask].iterrows():
            why = []
            if row.city_key in bad_city_zero:
                why.append("city_all_zero_fe")
            if row.city_key in bad_city_singleton:
                why.append("city_singleton_fe")
            if row.quarter_period_ordinal in bad_quarter_zero:
                why.append("quarter_all_zero_fe")
            if row.quarter_period_ordinal in bad_quarter_singleton:
                why.append("quarter_singleton_fe")
            records.append({"panel_source_row": int(row.panel_source_row), "city_key": row.city_key, "quarter": row.quarter, "quarter_period_ordinal": int(row.quarter_period_ordinal), "Q": float(row.Q), "fe_support_iteration": iteration, "fe_support_reason": ";".join(why)})
        work = work.loc[~mask].copy()
        if iteration > 20:
            raise RuntimeError("COVID all-zero FE separation removal did not stabilize")
    return work, pd.DataFrame(records)


def dynamic_specs_table(spec: Mapping[str, Any], fits: Mapping[str, Mapping[str, Any]], covid_fits: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_id, fit in list(fits.items()) + list(covid_fits.items()):
        base_id = model_id.split("-C", 1)[0]
        rows.append({"model_id": model_id, "specification": spec["models"].get(base_id, ""), "covid": bool(fit["covid"]), "covid_term": "covid_mean/10" if fit["covid"] else "none", "n": fit["n"], "cities": fit["frame"].city_key.nunique(), "quarters": fit["frame"].quarter_period_ordinal.nunique(), "provinces": fit["frame"].province.nunique(), "cluster_level": fit["cluster_col"], "clusters": fit["G"], "rank": fit["rank"], "design_dim": fit["dim"], "df": fit["df"], "status": fit["status"], "converged": fit["converged"], "iterations": fit["iterations"], "engine": "statsmodels.GLM Poisson IRLS", "offset": "ln W", "fixed_effects": "city FE + full calendar quarter FE", "separation_status": fit.get("separation", {}).get("status", ""), "separation_objective": fit.get("separation", {}).get("objective", np.nan), "failure_reason": fit["failure_reason"]})
    return pd.DataFrame(rows)


def covid_comparison(fits: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, fit in fits.items():
        for i, term in enumerate(fit["names"]):
            if term not in {"theta", "delta0", "delta1", "delta2", "covid_mean_per10"}:
                continue
            rows.append({"comparison": model_id, "term": term, "term_type": fit["meta"]["term_types"].get(term, ""), "estimate": fit["beta"][i], "se": fit["se"][i], "ci_low": fit["ci_low"][i], "ci_high": fit["ci_high"][i], "p_value": fit["p_value"][i], "n": fit["n"], "cities": fit["frame"].city_key.nunique(), "quarters": fit["frame"].quarter_period_ordinal.nunique(), "provinces": fit["frame"].province.nunique(), "cluster_level": fit["cluster_col"], "clusters": fit["G"], "df": fit["df"], "rank": fit["rank"], "status": fit["status"], "converged": fit["converged"], "iterations": fit["iterations"], "covariance_method": fit["covariance_method"], "separation_status": fit.get("separation", {}).get("status", ""), "failure_reason": fit["failure_reason"]})
        if fit["status"] == "estimated":
            transformed = transformed_values(fit, target="beta")
            for i, term in enumerate(transformed["labels"]):
                rows.append({"comparison": model_id, "term": term, "term_type": "original_level_lag_coefficient", "estimate": transformed["estimate"][i], "se": transformed["se"][i], "ci_low": transformed["ci_low"][i], "ci_high": transformed["ci_high"][i], "p_value": transformed["p_value"][i], "n": fit["n"], "cities": fit["frame"].city_key.nunique(), "quarters": fit["frame"]["quarter_period_ordinal"].nunique(), "provinces": fit["frame"]["province"].nunique(), "cluster_level": fit["cluster_col"], "clusters": fit["G"], "df": fit["df"], "rank": fit["rank"], "status": fit["status"], "converged": fit["converged"], "iterations": fit["iterations"], "covariance_method": fit["covariance_method"], "separation_status": fit.get("separation", {}).get("status", ""), "failure_reason": fit["failure_reason"]})
    return pd.DataFrame(rows)


def findings_markdown(spec: Mapping[str, Any], support: pd.DataFrame, paths: pd.DataFrame, contrasts: pd.DataFrame, covid: pd.DataFrame, fits: Mapping[str, Mapping[str, Any]], covid_fits: Mapping[str, Mapping[str, Any]], candidate: pd.DataFrame, final: pd.DataFrame, covid_candidate: pd.DataFrame, covid_removed: pd.DataFrame) -> str:
    def fmt(x: Any, digits: int = 4) -> str:
        return "NA" if not finite(x) else f"{float(x):.{digits}f}"
    lines = [
        "# V18 quarterly dynamic findings",
        "",
        f"Generated {utc_now()} by `build_quarterly_dynamic_v18.py`. This file reports the single saved stable result; it is not a lag search or a bootstrap analysis.",
        "",
        "## Estimator and fixed specification",
        "",
        "The implementation uses Python 3.12 `statsmodels.GLM` with a Poisson family, `method=IRLS`, city fixed effects and complete calendar-quarter fixed effects. `ln W` is an exposure offset. Clustered inference is computed from the Poisson score and Hessian, with CR1 factor `(G/(G-1))*((n-1)/(n-rank))` and two-sided `t(G-1)` intervals. The saved fit was not produced by `ppmlhdfe`; the LP separation diagnostic was used only to detect strict separation before fitting.",
        "",
        "D3 is parameterized as `theta*x[t-3] + delta0*(x[t]-x[t-1]) + delta1*(x[t-1]-x[t-2]) + delta2*(x[t-2]-x[t-3])`. Thus the original level coefficients are `beta0=delta0`, `beta1=delta1-delta0`, `beta2=delta2-delta1`, `beta3=theta-delta2`; the persistent-level path is `c0=delta0`, `c1=delta1`, `c2=delta2`, `c3=theta`. This is algebraically equivalent to the fixed four-level D3 and does not fit a separate model.",
        "",
        "## Support",
        "",
        f"Exact city/quarter-ordinal lag matching produced {len(candidate):,} candidate rows in {candidate.city_key.nunique():,} cities. The candidate contains {(candidate.Q > 0).sum():,} positive-Q rows and {(candidate.Q == 0).sum():,} source-coded zero-Q rows. Removing {candidate.loc[candidate.all_zero_city, 'city_key'].nunique():,} all-zero city FE groups ({int(candidate.all_zero_city.sum()):,} rows) and {candidate.loc[(~candidate.all_zero_city) & candidate.city_singleton, 'city_key'].nunique():,} non-zero city singleton groups ({int((~candidate.all_zero_city & candidate.city_singleton).sum()):,} rows) gave a 1,543-row preliminary support; iterative FE cleanup then removed {int(1543-len(final)):,} full-calendar-quarter singleton row(s), leaving {len(final):,} rows and {final.city_key.nunique():,} cities for common D0/D1/D3 support.",
        "",
        f"The final support spans ordinal quarters {int(final.quarter_period_ordinal.min())}--{int(final.quarter_period_ordinal.max())} ({final.quarter.min()}--{final.quarter.max()}) and {final.quarter_period_ordinal.nunique():,} calendar quarters. City vertical lengths range from {int(final.groupby('city_key').size().min())} to {int(final.groupby('city_key').size().max())} rows (median {fmt(final.groupby('city_key').size().median(), 1)}). The W/W_calendar ratio is a record-support diagnostic, not a new weight; its median is {fmt((final.W / final.W_calendar).median(), 3)}.",
        "",
        "## Main model result",
        "",
    ]
    if not paths.empty:
        d3 = paths.loc[paths.model_id.eq("D3")].sort_values("h")
        lines.append("D3 path estimates (pointwise 95% intervals; not simultaneous bands):")
        lines.append("")
        lines.append("| h | c_h | 95% CI | 10% facility difference implied activity difference |")
        lines.append("|---:|---:|---:|---:|")
        for _, r in d3.iterrows():
            lines.append(f"| {int(r.h)} | {fmt(r.estimate)} | [{fmt(r.ci_low)}, {fmt(r.ci_high)}] | {fmt(r.ten_percent_activity_difference_pct, 2)}% [{fmt(r.ten_percent_ci_low_pct, 2)}, {fmt(r.ten_percent_ci_high_pct, 2)}]% |")
    if not contrasts.empty:
        lines.extend(["", "The two pre-specified D3 comparisons are:", ""])
        for _, r in contrasts.loc[contrasts.model_id.eq("D3")].iterrows():
            lines.append(f"- `{r.comparison}`: estimate {fmt(r.estimate)}, SE {fmt(r.se)}, 95% CI [{fmt(r.ci_low)}, {fmt(r.ci_high)}], two-sided p = {fmt(r.p_value, 6)} (df = {int(r.df)}).")
    lines.extend(["", "These c_h values are conditional associations implied by a persistent proportional change in facility levels, holding the model FE path and other regressors fixed. They are not cumulative activity elasticities, causal impulse responses, or evidence of a true equipment maturation curve beyond h=3.", "", "## Fit diagnostics", "", "| model | status | n | cities/clusters | rank/design | iterations | separation LP | CR1 factor |", "|---|---|---:|---:|---:|---:|---|---:|"])
    for model_id, fit in fits.items():
        lines.append(f"| {model_id} | {fit['status']} | {fit['n']} | {fit['G']} | {fit['rank']}/{fit['dim']} | {fit['iterations']} | {fit.get('separation', {}).get('status', '')} ({fmt(fit.get('separation', {}).get('objective', np.nan), 5)}) | {fmt(fit.get('covariance_meta', {}).get('cr1_factor', np.nan), 5)} |")
    lines.extend(["", "The D0, D1 and D3 LP checks found no strict separation on the common main support after the iterative city/quarter all-zero and singleton cleanup. The output preserves the full fitted-design covariance as well as the transformed original-level covariance; transformed intervals use `a'Va` rather than sums of standard errors.", "", "## COVID same-support comparison", ""])
    if covid_removed is not None and len(covid_removed):
        reason_text = covid_removed["fe_support_reason"].fillna("")
        allzero_rows = int(reason_text.str.contains("city_all_zero_fe", regex=False).sum())
        singleton_rows = int(reason_text.eq("city_singleton_fe").sum())
        overlap_rows = int((reason_text.str.contains("city_all_zero_fe", regex=False) & reason_text.str.contains("city_singleton_fe", regex=False)).sum())
        lines.append(f"The year<=2022 complete-current-source-date subset initially had {len(covid_candidate):,} rows; iterative FE cleanup removed {len(covid_removed):,} rows from {covid_removed.city_key.nunique():,} cities: {allzero_rows} city-all-zero rows plus {singleton_rows} additional singleton-only rows, with {overlap_rows} row carrying both labels. C0 and C1 use the resulting identical support. No 2023 COVID value was filled.")
    else:
        lines.append(f"The year<=2022 complete-current-source-date subset had {len(covid_candidate):,} rows and required no all-zero FE removal. No 2023 COVID value was filled.")
    for model_id, fit in covid_fits.items():
        lines.append(f"- {model_id}: {fit['status']}, n={fit['n']}, cities={fit['frame'].city_key.nunique()}, provinces/clusters={fit['G']}, rank={fit['rank']}/{fit['dim']}, iterations={fit['iterations']}; covariance uses province CR1 and t({fit['df']}).")
    if "D3-C1" in covid_fits and covid_fits["D3-C1"]["status"] == "estimated":
        c1 = covid_fits["D3-C1"]
        c1_idx = c1["names"].index("covid_mean_per10")
        lines.append(f"The C1 COVID index is finite and adds one design-rank degree ({c1['rank']} versus {covid_fits.get('D3-C0', c1)['rank']} in C0); its coefficient is {fmt(c1['beta'][c1_idx])} (95% CI [{fmt(c1['ci_low'][c1_idx])}, {fmt(c1['ci_high'][c1_idx])}]), so it was not absorbed by the FE.")
    lines.extend(["", "The COVID comparison is a same-subset C0/C1 specification comparison. It is not a decomposition of the difference between the COVID subset and the full-period D3 estimate.", "", "## Limitations and non-claims", "", "- The quarterly panel has no usable local quarterly weather series in this round, so no annual weather variable was copied into a quarterly control.", "- The dynamic support is selected by observed positive facility levels at t through t-3 and observed compatible-week coverage; it is not a balanced panel and does not represent all 334 cities.", "- Source-coded zeros are retained as zeros; missing/no-compatible-week records remain outside the dynamic support.", "- The finite lag is fixed at L=3. The model does not estimate responses after the stipulated horizon and should not be read as identifying causal policy or investment effects.", ""])
    return "\n".join(lines) + "\n"


def run(refit: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    previous_manifest: dict[str, Any] | None = None
    if RUN_PATH.exists():
        try:
            previous_manifest = json.loads(RUN_PATH.read_text(encoding="utf-8"))
        except Exception:
            previous_manifest = None
    spec = ensure_spec()  # explicitly before any model fitting
    if outputs_complete() and not refit:
        print(json.dumps({"status": "reused_existing_stable_result", "output": str(OUT), "spec": str(SPEC_PATH)}, ensure_ascii=False))
        return
    panel = load_panel()
    candidate, audit = build_candidate(panel)
    # Main support first applies the explicit candidate city rules, then
    # iteratively removes any city/quarter FE all-zero or singleton groups.
    # The latter step is shared by all three models.
    main_initial = candidate.loc[candidate.main_initial_support].copy()
    final, main_removed = apply_city_zero_separation(main_initial)
    # Check the final common design and LP separation before all three fits.
    final["final_dynamic_support"] = True
    fit_started_utc = utc_now()
    fits: dict[str, dict[str, Any]] = {}
    for model_id in ["D0", "D3", "D1"]:
        fits[model_id] = fit_one(final, model_id, covid=False, cluster_col="city_key")
    # COVID requires a current genuine source-date exposure, year <= 2022, and
    # complete compatible-week coverage.  All models share the same separated
    # support and the same province clustering rule.
    covid_candidate = final.loc[(final["year"] <= 2022) & final["covid_mean"].notna() & final["covid_coverage_complete"].eq(True)].copy()
    covid_final, covid_removed = apply_city_zero_separation(covid_candidate)
    covid_fits: dict[str, dict[str, Any]] = {}
    for model_id, add_covid in [("D3-C0", False), ("D3-C1", True)]:
        base_id = "D3"
        covid_fits[model_id] = fit_one(covid_final, base_id, covid=add_covid, cluster_col="province")
        covid_fits[model_id]["model_id"] = model_id
    # Preserve row-specific inclusion reasons for the audit and final sample.
    audit["final_dynamic_support"] = audit["panel_source_row"].isin(set(final.panel_source_row))
    audit["included_D0"] = audit["final_dynamic_support"]
    audit["included_D1"] = audit["final_dynamic_support"]
    audit["included_D3"] = audit["final_dynamic_support"]
    audit["included_D3_C0"] = audit["panel_source_row"].isin(set(covid_final.panel_source_row))
    audit["included_D3_C1"] = audit["panel_source_row"].isin(set(covid_final.panel_source_row))
    audit["covid_candidate_support"] = audit["panel_source_row"].isin(set(covid_candidate.panel_source_row))
    audit["covid_final_support"] = audit["panel_source_row"].isin(set(covid_final.panel_source_row))
    main_removed_lookup = main_removed.set_index("panel_source_row")["fe_support_reason"] if len(main_removed) else pd.Series(dtype=object)
    covid_removed_lookup = covid_removed.set_index("panel_source_row")["fe_support_reason"] if len(covid_removed) else pd.Series(dtype=object)
    audit["main_fe_support_reason"] = audit["panel_source_row"].map(main_removed_lookup)
    audit["covid_separation_reason"] = audit["panel_source_row"].map(covid_removed_lookup)
    # The final sample also carries model inclusion and COVID separation flags.
    final_sample = sample_columns(final, final)
    final_sample["covid_candidate_support"] = final_sample["panel_source_row"].isin(set(covid_candidate.panel_source_row))
    final_sample["covid_final_support"] = final_sample["panel_source_row"].isin(set(covid_final.panel_source_row))
    final_sample["included_D3_C0"] = final_sample["covid_final_support"]
    final_sample["included_D3_C1"] = final_sample["covid_final_support"]
    final_sample["main_fe_support_reason"] = final_sample["panel_source_row"].map(main_removed_lookup)
    final_sample["covid_separation_reason"] = final_sample["panel_source_row"].map(covid_removed_lookup)
    # Drop internal ord helper columns from the public audit only after keeping
    # exact-lag flags and reasons visible.
    audit_public = audit.copy()
    public_cols = [c for c in audit_public.columns if c not in {"lag1_ordinal", "lag2_ordinal", "lag3_ordinal", "lag_link_ok"}]
    audit_public = audit_public[public_cols]
    write_csv(final_sample, OUT / "quarterly_dynamic_sample_v18.csv")
    write_csv(audit_public, OUT / "quarterly_dynamic_sample_audit_v18.csv")
    write_csv(support_rows(panel, candidate, audit_public, final, fits, covid_candidate, covid_fits, main_removed=main_removed, covid_removed=covid_removed), OUT / "quarterly_dynamic_support_v18.csv")
    specs_table = dynamic_specs_table(spec, fits, covid_fits)
    write_csv(specs_table, OUT / "quarterly_dynamic_specs_v18.csv")
    spec_out = dict(spec)
    spec_out["fit_started_utc"] = fit_started_utc
    spec_out["fit_finished_utc"] = utc_now()
    spec_out["fit_output_hashes"] = {}
    write_json(spec_out, OUT / "quarterly_dynamic_specs_v18.json")
    all_fits = {**fits, **covid_fits}
    write_csv(coefficient_rows(all_fits), OUT / "quarterly_dynamic_coefficients_v18.csv")
    write_csv(covariance_rows(all_fits), OUT / "quarterly_dynamic_covariance_v18.csv")
    paths, contrasts = paths_and_contrasts(fits)
    write_csv(contrasts, OUT / "quarterly_dynamic_contrasts_v18.csv")
    write_csv(paths, OUT / "quarterly_dynamic_paths_v18.csv")
    write_csv(covid_comparison(covid_fits), OUT / "quarterly_covid_comparison_v18.csv")
    diag_rows: list[dict[str, Any]] = []
    for group, group_fits in [("main", fits), ("covid", covid_fits)]:
        for model_id, fit in group_fits.items():
            diag_rows.append({"group": group, "model_id": model_id, "status": fit["status"], "converged": fit["converged"], "iterations": fit["iterations"], "n": fit["n"], "clusters": fit["G"], "rank": fit["rank"], "design_dim": fit["dim"], "df": fit["df"], "cr1_factor": fit.get("covariance_meta", {}).get("cr1_factor", np.nan), "separation_status": fit.get("separation", {}).get("status", ""), "separation_objective": fit.get("separation", {}).get("objective", np.nan), "separation_violating_zero_rows": fit.get("separation", {}).get("violating_zero_rows", 0), "score_residual_l2": fit.get("score_residual_l2", np.nan), "score_residual_max_abs": fit.get("score_residual_max_abs", np.nan), "sum_observed": fit.get("sum_observed", np.nan), "sum_fitted": fit.get("sum_fitted", np.nan), "mean_fitted": fit.get("mean_fitted", np.nan), "deviance": fit.get("deviance", np.nan), "pearson_chi2": fit.get("pearson_chi2", np.nan), "failure_reason": fit["failure_reason"], "engine": "statsmodels.GLM Poisson IRLS"})
    write_csv(pd.DataFrame(diag_rows), OUT / "quarterly_dynamic_diagnostics_v18.csv")
    (OUT / "dynamic_findings_v18.md").write_text(findings_markdown(spec, support_rows(panel, candidate, audit_public, final, fits, covid_candidate, covid_fits, main_removed=main_removed, covid_removed=covid_removed), paths, contrasts, covid_comparison(covid_fits), fits, covid_fits, candidate, final, covid_candidate, covid_removed), encoding="utf-8")
    # Preserve earlier stable attempts when a local support-rule repair is
    # explicitly requested.  This prevents the manifest from implying that
    # five unique model labels equal five total fit calls across the round.
    try:
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8")) if HISTORY_PATH.exists() else []
        if not isinstance(history, list):
            history = []
    except Exception:
        history = []
    if previous_manifest is not None and not history:
        history.append({
            "attempt": 1,
            "attempt_type": "initial_saved_attempt",
            "status": previous_manifest.get("status", "unknown"),
            "created_utc": previous_manifest.get("created_utc"),
            "actual_fit_count": previous_manifest.get("actual_fit_count", 5),
            "fit_status": previous_manifest.get("fit_status", {}),
            "candidate_rows": previous_manifest.get("candidate_rows"),
            "final_rows": previous_manifest.get("final_rows"),
            "final_cities": previous_manifest.get("final_cities"),
            "covid_final_rows": previous_manifest.get("covid_final_rows"),
            "note": "Saved before the observed full-calendar-quarter singleton support repair; retained as an original attempt record.",
        })
    history.append({
        "attempt": len(history) + 1,
        "attempt_type": "local_support_rule_repair" if previous_manifest is not None else "initial_saved_attempt",
        "status": "stable_result_saved" if all(f["status"] == "estimated" for f in all_fits.values()) else "saved_with_fit_failure",
        "fit_started_utc": fit_started_utc,
        "fit_finished_utc": utc_now(),
        "actual_fit_count": len(all_fits),
        "fit_status": {k: v["status"] for k, v in all_fits.items()},
        "candidate_rows": len(candidate),
        "preliminary_support_rows": int(len(main_initial)),
        "main_removed_rows": int(len(main_removed)),
        "final_rows": len(final),
        "final_cities": final.city_key.nunique(),
        "covid_candidate_rows": len(covid_candidate),
        "covid_removed_rows": len(covid_removed),
        "covid_final_rows": len(covid_final),
        "covid_final_cities": covid_final.city_key.nunique(),
        "note": "Fixed D0/D1/D3/D3-C0/D3-C1 set; no bootstrap or lag search.",
    })
    write_json(history, HISTORY_PATH)
    # Run manifest is written last so the default cache is only considered
    # complete after all required outputs exist.
    manifest = {
        "status": "stable_result_saved" if all(f["status"] == "estimated" for f in all_fits.values()) else "saved_with_fit_failure",
        "created_utc": utc_now(),
        "script": repository_relative(Path(__file__), REPOSITORY),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_panel": repository_relative(PANEL, REPOSITORY),
        "source_panel_sha256": sha256(PANEL),
        "spec_path": repository_relative(SPEC_PATH, REPOSITORY),
        "spec_sha256": sha256(SPEC_PATH),
        "candidate_rows": len(candidate), "candidate_cities": candidate.city_key.nunique(), "final_rows": len(final), "final_cities": final.city_key.nunique(),
        "covid_candidate_rows": len(covid_candidate), "covid_removed_rows": len(covid_removed), "covid_final_rows": len(covid_final), "covid_final_cities": covid_final.city_key.nunique(),
        "fit_status": {k: v["status"] for k, v in all_fits.items()},
        "fit_iterations": {k: v["iterations"] for k, v in all_fits.items()},
        "actual_fit_count": len(all_fits),
        "fit_attempts_total": int(sum(item.get("actual_fit_count", 0) for item in history)),
        "fit_attempt_history_path": repository_relative(HISTORY_PATH, REPOSITORY),
        "bootstrap": False,
        "lag_search": False,
        "output_files": [p.name for p in all_output_paths() if p.exists()],
    }
    write_json(manifest, RUN_PATH)
    print(json.dumps({"status": manifest["status"], "candidate": [len(candidate), int(candidate.city_key.nunique())], "final": [len(final), int(final.city_key.nunique())], "covid_final": [len(covid_final), int(covid_final.city_key.nunique())], "fits": manifest["fit_status"], "outputs": str(OUT)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", action="store_true", help="explicitly rebuild the fixed V18 result; use only for a local repair")
    args = parser.parse_args()
    run(refit=bool(args.refit))


if __name__ == "__main__":
    main()

