"""
Next-Best-Action Recommendation Engine
=======================================
Collaborative filtering + business rules to surface the most relevant
offer / action for each customer based on their behavioral signal.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


# ── Sample Data Schema ────────────────────────────────────────────────────
# In production: load from PostgreSQL
# SELECT customer_id, product_id, interaction_score FROM customer_events


def build_user_item_matrix(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a user-item matrix from interaction events.

    interaction_score: implicit feedback (view=1, add_to_cart=3, purchase=5)
    """
    matrix = interactions_df.pivot_table(
        index="customer_id",
        columns="product_id",
        values="interaction_score",
        aggfunc="sum",
        fill_value=0,
    )
    return matrix


def get_similar_customers(
    matrix: pd.DataFrame,
    customer_id: str,
    top_n: int = 10,
) -> list[str]:
    """Find the N most similar customers using cosine similarity."""
    if customer_id not in matrix.index:
        return []

    similarity = cosine_similarity(matrix)
    sim_df = pd.DataFrame(
        similarity, index=matrix.index, columns=matrix.index
    )

    similar = (
        sim_df[customer_id]
        .drop(customer_id)          # exclude self
        .nlargest(top_n)
        .index.tolist()
    )
    return similar


def get_recommendations(
    matrix: pd.DataFrame,
    customer_id: str,
    top_k: int = 5,
    exclude_purchased: bool = True,
) -> list[dict]:
    """
    Generate top-K next-best-action recommendations.

    Parameters
    ----------
    matrix        : user-item interaction matrix
    customer_id   : target customer
    top_k         : number of recommendations to return
    exclude_purchased : exclude items the customer already interacted with

    Returns
    -------
    list of {"product_id": ..., "score": ..., "reason": ...}
    """
    if customer_id not in matrix.index:
        return _cold_start_recommendations(top_k)

    similar_customers = get_similar_customers(matrix, customer_id)

    if not similar_customers:
        return _cold_start_recommendations(top_k)

    # Aggregate scores from similar customers (weighted by similarity)
    sim_scores = cosine_similarity(
        matrix.loc[[customer_id]], matrix.loc[similar_customers]
    )[0]

    weighted_scores = (
        matrix.loc[similar_customers]
        .multiply(sim_scores, axis=0)
        .sum()
    )

    # Exclude already-interacted items
    if exclude_purchased:
        already_interacted = matrix.loc[customer_id]
        already_interacted = already_interacted[already_interacted > 0].index
        weighted_scores = weighted_scores.drop(already_interacted, errors="ignore")

    top_products = (
        weighted_scores
        .nlargest(top_k)
        .reset_index()
    )
    top_products.columns = ["product_id", "score"]

    recommendations = []
    for _, row in top_products.iterrows():
        recommendations.append({
            "product_id": row["product_id"],
            "score":      round(row["score"], 4),
            "reason":     f"Customers similar to you also loved this product.",
        })

    return recommendations


def _cold_start_recommendations(top_k: int) -> list[dict]:
    """Fallback for new customers (cold start problem)."""
    # In production: return trending / bestseller products from DB
    bestsellers = [f"PROD-{i:03d}" for i in range(1, top_k + 1)]
    return [
        {"product_id": pid, "score": 1.0 - (i * 0.05), "reason": "Trending this week"}
        for i, pid in enumerate(bestsellers)
    ]


def hit_rate_at_k(
    test_interactions: pd.DataFrame,
    matrix: pd.DataFrame,
    k: int = 5,
) -> float:
    """
    Evaluate recommendation quality using Hit Rate @ K.
    Hit Rate = fraction of test users where at least 1 relevant item is in top-K.
    """
    hits = 0
    total = 0

    for customer_id, group in test_interactions.groupby("customer_id"):
        actual_items = set(group["product_id"].tolist())
        recs = get_recommendations(matrix, customer_id, top_k=k)
        rec_items = set(r["product_id"] for r in recs)

        if actual_items & rec_items:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0


# ── Demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic interaction data
    np.random.seed(42)
    n_customers = 50
    n_products  = 20

    data = []
    for cust in [f"CUST-{i:03d}" for i in range(n_customers)]:
        for prod in np.random.choice([f"PROD-{j:03d}" for j in range(n_products)], size=6):
            score = np.random.choice([1, 3, 5], p=[0.5, 0.3, 0.2])
            data.append({"customer_id": cust, "product_id": prod, "interaction_score": score})

    df      = pd.DataFrame(data)
    matrix  = build_user_item_matrix(df)

    recs = get_recommendations(matrix, "CUST-001", top_k=5)
    print("\n🎯 Top-5 Next-Best-Action for CUST-001:")
    for r in recs:
        print(f"  • {r['product_id']} | Score: {r['score']:.3f} | {r['reason']}")

    hr = hit_rate_at_k(df.sample(frac=0.2, random_state=42), matrix, k=5)
    print(f"\n📊 Hit Rate @5: {hr:.2%}")
