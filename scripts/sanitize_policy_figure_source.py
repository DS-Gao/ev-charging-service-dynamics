"""Remove restricted policy text and local source paths from figure-source CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
FIGURE_SOURCE = REPOSITORY / "data" / "figure_source"
PATH_COLUMNS = {
    "g_registry_first_path",
    "g_registry_g_first_path",
    "first_B_source",
    "document_source_text_file",
    "g_document_source_text_file",
    "earliest_B_source_text_file",
    "source_text_file",
    "source_locator",
    "example_source_locator",
    "source",
}


def _public_source(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value)
    if "Raw data" in text or "outputs/" in text or "analysis/" in text or ":\\" in text or ":/" in text:
        return "metadata/policy_sources.csv"
    return value


def main() -> None:
    # Long normative quotations are not needed to draw the figure. They are
    # intentionally omitted from the public copy; source keys and numeric
    # coding columns remain available for lookup in metadata/policy_sources.csv.
    responsibility = FIGURE_SOURCE / "fig8_responsibility_clauses_v20.csv"
    frame = pd.read_csv(responsibility, encoding="utf-8-sig", low_memory=False)
    frame = frame.drop(columns=[c for c in ("quote", "context") if c in frame.columns])
    for column in PATH_COLUMNS.intersection(frame.columns):
        frame[column] = frame[column].map(_public_source)
    frame.to_csv(responsibility, index=False, encoding="utf-8-sig")

    for name in ("fig8_document_sequence_v20.csv", "fig8_policy_categories_v20.csv", "figS_governance_timing_support_v20.csv"):
        path = FIGURE_SOURCE / name
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        for column in PATH_COLUMNS.intersection(frame.columns):
            frame[column] = frame[column].map(_public_source)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(path.relative_to(REPOSITORY))

    audit = REPOSITORY / "data" / "results" / "reference" / "policy" / "covid_model_sample_audit_v11.csv"
    frame = pd.read_csv(audit, encoding="utf-8-sig", low_memory=False)
    drop = [c for c in frame.columns if "source_path" in c.lower() or "absolute" in c.lower()]
    if drop:
        frame = frame.drop(columns=drop)
    frame.to_csv(audit, index=False, encoding="utf-8-sig")
    print(audit.relative_to(REPOSITORY), "dropped", len(drop), "local path columns")


if __name__ == "__main__":
    main()
