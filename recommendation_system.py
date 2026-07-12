"""Content-based Coursera course recommender for DS 423 Group 16.

Run:
    python recommendation_system.py --data Coursera_cleaned_for_recommendation.csv

The script prints example recommendations and writes reproducible evaluation
artifacts to ``outputs/``. No learner interaction data is present, so evaluation
uses skill-overlap relevance as an explicit offline proxy.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


RANDOM_STATE = 42


def load_courses(path: str | Path) -> pd.DataFrame:
    """Load and minimally validate the cleaned course catalog."""
    df = pd.read_csv(path)
    required = {
        "course_id", "course_title", "university", "difficulty_level",
        "course_rating_filled", "course_url", "course_description", "skills",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    text_cols = ["course_title", "university", "difficulty_level",
                 "course_description", "skills"]
    df[text_cols] = df[text_cols].fillna("")
    df = df.drop_duplicates("course_id").reset_index(drop=True)
    return df


def build_text(df: pd.DataFrame) -> pd.Series:
    """Weight short, informative metadata by repeating it before TF-IDF."""
    return (
        (df["course_title"] + " ") * 3
        + (df["skills"] + " ") * 2
        + df["university"] + " "
        + df["difficulty_level"] + " "
        + df["course_description"]
    ).str.lower()


def fit_model(df: pd.DataFrame, max_features: int = 30_000):
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2,
        max_df=0.95, max_features=max_features, sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(build_text(df))
    return vectorizer, matrix


def recommend_by_index(
    df: pd.DataFrame, matrix, index: int, n: int = 10,
    rating_weight: float = 0.05,
) -> pd.DataFrame:
    """Return top-N similar courses, with a small rating tie-breaker."""
    similarity = linear_kernel(matrix[index], matrix).ravel()
    rating = df["course_rating_filled"].astype(float).to_numpy()
    rating_scaled = np.clip((rating - 1.0) / 4.0, 0, 1)
    score = (1 - rating_weight) * similarity + rating_weight * rating_scaled
    score[index] = -np.inf
    top = np.argpartition(score, -n)[-n:]
    top = top[np.argsort(score[top])[::-1]]
    result = df.loc[top, ["course_id", "course_title", "university",
                          "difficulty_level", "course_rating_filled", "course_url"]].copy()
    result.insert(0, "rank", range(1, len(result) + 1))
    result["score"] = score[top].round(4)
    return result.reset_index(drop=True)


def recommend_by_title(df: pd.DataFrame, matrix, title: str, n: int = 10):
    """Recommend from the closest case-insensitive title match."""
    exact = df.index[df["course_title"].str.casefold() == title.casefold()].tolist()
    if exact:
        idx = exact[0]
    else:
        contains = df.index[df["course_title"].str.contains(
            re.escape(title), case=False, regex=True
        )].tolist()
        if not contains:
            raise KeyError(f"Course title not found: {title}")
        idx = contains[0]
    return recommend_by_index(df, matrix, idx, n=n)


def skill_sets(df: pd.DataFrame) -> list[set[str]]:
    return [
        {s.strip().lower() for s in str(value).split(",") if s.strip()}
        for value in df["skills"]
    ]


def evaluate(
    df: pd.DataFrame, matrix, ks=(5, 10, 20), sample_size=250,
    relevance_threshold=0.20,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate retrieval against a transparent skill-Jaccard proxy.

    A catalog item is relevant when its skill-set Jaccard similarity to the
    query is at least ``relevance_threshold``. Queries without any relevant
    neighbors are excluded, and the count is reported.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    skills = skill_sets(df)
    candidates = [i for i, s in enumerate(skills) if len(s) >= 2]
    query_ids = rng.choice(candidates, size=min(sample_size, len(candidates)), replace=False)
    rows, example = [], {}
    max_k = max(ks)
    for q in query_ids:
        a = skills[q]
        relevance = np.zeros(len(df), dtype=bool)
        for j, b in enumerate(skills):
            if j != q and a and b:
                relevance[j] = len(a & b) / len(a | b) >= relevance_threshold
        relevant_count = int(relevance.sum())
        if relevant_count == 0:
            continue
        scores = linear_kernel(matrix[q], matrix).ravel()
        scores[q] = -np.inf
        ranked = np.argpartition(scores, -max_k)[-max_k:]
        ranked = ranked[np.argsort(scores[ranked])[::-1]]
        for k in ks:
            hits = int(relevance[ranked[:k]].sum())
            rows.append({"query": q, "k": k, "precision": hits / k,
                         "recall": hits / relevant_count, "hits": hits,
                         "relevant": relevant_count})
        if not example:
            example = {"query_index": int(q), "query_title": df.at[q, "course_title"],
                       "relevant_count": relevant_count}
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No evaluable queries; lower the relevance threshold.")
    summary = detail.groupby("k", as_index=False).agg(
        precision_at_k=("precision", "mean"),
        recall_at_k=("recall", "mean"),
        evaluated_queries=("query", "nunique"),
    )
    return summary, {**example, "sampled_queries": len(query_ids),
                     "evaluated_queries": int(detail["query"].nunique()),
                     "relevance_threshold": relevance_threshold}


def save_charts(df: pd.DataFrame, metrics: pd.DataFrame, out: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#1B4965", "#5FA8D3"]
    ax = metrics.set_index("k")[["precision_at_k", "recall_at_k"]].plot(
        kind="bar", figsize=(8, 4.6), color=colors, width=0.72
    )
    ax.set(title="Offline Retrieval Performance", xlabel="K", ylabel="Mean score", ylim=(0, 1))
    ax.legend(["Precision@K", "Recall@K"], frameon=False)
    plt.tight_layout(); plt.savefig(out / "evaluation_metrics.png", dpi=180); plt.close()

    counts = df["difficulty_level"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    counts.plot(kind="bar", ax=ax, color="#1B4965")
    ax.set(title="Courses by Difficulty Level", xlabel="", ylabel="Number of courses")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig(out / "difficulty_distribution.png", dpi=180); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Coursera_cleaned_for_recommendation.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df = load_courses(args.data)
    vectorizer, matrix = fit_model(df)
    metrics, protocol = evaluate(df, matrix)
    metrics.to_csv(out / "evaluation_metrics.csv", index=False)
    sample_index = int(np.argmax(df["course_title"].str.contains(
        "machine learning", case=False, regex=False).to_numpy()))
    recs = recommend_by_index(df, matrix, sample_index, n=args.top_n)
    recs.to_csv(out / "sample_top10_recommendations.csv", index=False)
    save_charts(df, metrics, out)
    metadata = {
        "rows": len(df), "columns": len(df.columns), "tfidf_features": matrix.shape[1],
        "matrix_nonzeros": int(matrix.nnz), "query_course": df.at[sample_index, "course_title"],
        "protocol": protocol,
    }
    (out / "run_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print("\nQuery:", metadata["query_course"])
    print(recs[["rank", "course_title", "score"]].to_string(index=False))
    print(f"\nArtifacts written to {out.resolve()}")


if __name__ == "__main__":
    main()
