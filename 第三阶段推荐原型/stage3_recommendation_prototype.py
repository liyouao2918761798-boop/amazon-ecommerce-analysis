
import os
import re
import math
import random
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/mnt/data"
OUT_DIR = os.path.join(DATA_DIR, "stage3_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_catalog():
    catalog = pd.read_csv(os.path.join(DATA_DIR, "stage1_outputs", "clean_amazon_catalog.csv"))
    catalog["full_review_text_raw"] = (
        catalog["review_title"].fillna("") + ". " + catalog["review_content"].fillna("")
    ).str.strip()
    catalog["full_review_text"] = (
        catalog["full_review_text_raw"]
        .str.replace(r"https?://\S+", " ", regex=True)
        .str.replace(r"[^A-Za-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )
    return catalog


def run_textblob_sentiment(catalog):
    def tb_metrics(text):
        if not isinstance(text, str) or not text.strip():
            return (np.nan, np.nan)
        sentiment = TextBlob(text).sentiment
        return (sentiment.polarity, sentiment.subjectivity)

    metrics = catalog["full_review_text"].apply(tb_metrics)
    catalog["text_polarity"] = metrics.apply(lambda x: x[0])
    catalog["text_subjectivity"] = metrics.apply(lambda x: x[1])

    def label5(p):
        if pd.isna(p):
            return "NoText"
        if p > 0.30:
            return "Strong Positive"
        if p > 0.10:
            return "Mild Positive"
        if p >= -0.10:
            return "Neutral"
        if p >= -0.30:
            return "Mild Negative"
        return "Strong Negative"

    def label3(p):
        if pd.isna(p):
            return "NoText"
        if p > 0.10:
            return "Positive"
        if p < -0.10:
            return "Negative"
        return "Neutral"

    def rating_label(r):
        if pd.isna(r):
            return "Unknown"
        if r >= 4:
            return "Positive"
        if r <= 2:
            return "Negative"
        return "Neutral"

    catalog["sentiment_label_5"] = catalog["text_polarity"].apply(label5)
    catalog["sentiment_label_3"] = catalog["text_polarity"].apply(label3)
    catalog["rating_label_3"] = catalog["rating_num"].apply(rating_label)
    catalog["sentiment_rating_alignment"] = np.where(
        catalog["sentiment_label_3"] == catalog["rating_label_3"], "Aligned", "Mismatched"
    )
    catalog["polarity_gap_abs"] = np.abs(catalog["text_polarity"] - ((catalog["rating_num"] - 3) / 2))
    return catalog


def run_lda(catalog, n_topics=5):
    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.65,
        min_df=8,
        ngram_range=(1, 2),
        max_features=2500,
    )
    X = vectorizer.fit_transform(catalog["full_review_text"].fillna(""))
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
        max_iter=30,
    )
    doc_topic = lda.fit_transform(X)
    terms = vectorizer.get_feature_names_out()

    topic_label_map = {
        0: "Device features, battery and display",
        1: "Ease of use and everyday usability",
        2: "Price-value and general satisfaction",
        3: "Audio, TV and peripheral performance",
        4: "Charging and cable performance",
    }

    topic_rows = []
    for topic_idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:12]
        top_terms = [terms[i] for i in top_idx]
        topic_rows.append(
            {
                "topic_id": int(topic_idx),
                "topic_label": topic_label_map.get(topic_idx, f"Topic {topic_idx}"),
                "top_terms": ", ".join(top_terms),
            }
        )

    catalog["dominant_topic_id"] = doc_topic.argmax(axis=1)
    catalog["dominant_topic_label"] = catalog["dominant_topic_id"].map(topic_label_map)
    catalog["topic_probability"] = doc_topic.max(axis=1)

    return catalog, pd.DataFrame(topic_rows)


def load_rating_sample(max_chunks=10, chunksize=200000):
    chunks = []
    zip_path = os.path.join(DATA_DIR, "ratings_Electronics (1).csv.zip")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("ratings_Electronics (1).csv") as fp:
            reader = pd.read_csv(
                fp,
                names=["user_id", "product_id", "rating", "timestamp"],
                chunksize=chunksize,
            )
            for idx, chunk in enumerate(reader):
                chunks.append(chunk)
                if idx + 1 >= max_chunks:
                    break
    return pd.concat(chunks, ignore_index=True)


def build_association_recommender(ratings):
    positive = ratings.loc[ratings["rating"] >= 4, ["user_id", "product_id"]].drop_duplicates().copy()

    user_counts = positive.groupby("user_id")["product_id"].nunique()
    item_counts = positive.groupby("product_id")["user_id"].nunique()

    filtered = positive[
        positive["user_id"].isin(user_counts[user_counts >= 5].index)
        & positive["product_id"].isin(item_counts[item_counts >= 20].index)
    ].copy()

    # final active-user filter
    user_counts_2 = filtered.groupby("user_id")["product_id"].nunique()
    filtered = filtered[filtered["user_id"].isin(user_counts_2[user_counts_2 >= 5].index)].copy()

    user_items = filtered.groupby("user_id")["product_id"].apply(list).to_dict()

    rng = random.Random(42)
    train_baskets, test_baskets = {}, {}
    for user_id, items in user_items.items():
        items = list(set(items))
        rng.shuffle(items)
        holdout_n = max(1, int(round(len(items) * 0.2)))
        holdout_n = min(holdout_n, len(items) - 3)
        if holdout_n < 1:
            continue
        test_items = items[:holdout_n]
        train_items = items[holdout_n:]
        if len(train_items) < 3:
            continue
        train_baskets[user_id] = train_items
        test_baskets[user_id] = test_items

    item_support = Counter()
    cooc = defaultdict(Counter)

    for items in train_baskets.values():
        uniq = list(dict.fromkeys(items))
        for item in uniq:
            item_support[item] += 1
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                cooc[a][b] += 1
                cooc[b][a] += 1

    def recommend(train_items, k=10):
        scores = Counter()
        seen = set(train_items)
        for item in seen:
            base_pop = item_support.get(item, 1)
            for candidate, cij in cooc.get(item, {}).items():
                if candidate in seen:
                    continue
                score = (cij / base_pop) + (cij / math.sqrt(base_pop * item_support.get(candidate, 1)))
                scores[candidate] += score

        if not scores:
            ranked = [item for item, _ in item_support.most_common() if item not in seen][:k]
        else:
            ranked = [item for item, _ in scores.most_common(k * 5) if item not in seen][:k]
            if len(ranked) < k:
                ranked.extend([item for item, _ in item_support.most_common() if item not in seen and item not in ranked][: k - len(ranked)])
        return ranked[:k]

    return filtered, train_baskets, test_baskets, item_support, cooc, recommend


def evaluate_recommender(train_baskets, test_baskets, item_support, recommend):
    popular_items = [item for item, _ in item_support.most_common()]

    def popularity_recommend(train_items, k=10):
        seen = set(train_items)
        return [item for item in popular_items if item not in seen][:k]

    active_eval_users = [u for u, items in train_baskets.items() if len(items) + len(test_baskets[u]) >= 8]

    result_rows = []
    baseline_rows = []

    for user_id in active_eval_users:
        recs = recommend(train_baskets[user_id], k=10)
        truth = set(test_baskets[user_id])
        hits = len(truth.intersection(recs))
        result_rows.append(
            {
                "user_id": user_id,
                "precision_at_10": hits / 10,
                "recall_at_10": hits / len(truth),
                "hit_rate_at_10": 1 if hits > 0 else 0,
            }
        )

        baseline_recs = popularity_recommend(train_baskets[user_id], 10)
        baseline_hits = len(truth.intersection(baseline_recs))
        baseline_rows.append(
            {
                "user_id": user_id,
                "precision_at_10": baseline_hits / 10,
                "recall_at_10": baseline_hits / len(truth),
                "hit_rate_at_10": 1 if baseline_hits > 0 else 0,
            }
        )

    return pd.DataFrame(result_rows), pd.DataFrame(baseline_rows)


if __name__ == "__main__":
    catalog = load_catalog()
    catalog = run_textblob_sentiment(catalog)
    catalog, topic_terms = run_lda(catalog)

    rating_sample = load_rating_sample(max_chunks=10, chunksize=200000)
    filtered, train_baskets, test_baskets, item_support, cooc, recommender = build_association_recommender(rating_sample)
    assoc_eval, pop_eval = evaluate_recommender(train_baskets, test_baskets, item_support, recommender)

    topic_terms.to_csv(os.path.join(OUT_DIR, "lda_topic_terms_summary.csv"), index=False)
    assoc_eval.to_csv(os.path.join(OUT_DIR, "recommendation_eval_user_level.csv"), index=False)

    metrics = pd.DataFrame(
        [
            {
                "model": "Association-based recommender",
                "evaluation_users": len(assoc_eval),
                "source_rows_used": len(rating_sample),
                "positive_interactions_used": len(filtered),
                "precision_at_10": assoc_eval["precision_at_10"].mean(),
                "recall_at_10": assoc_eval["recall_at_10"].mean(),
                "accuracy_hit_rate_at_10": assoc_eval["hit_rate_at_10"].mean(),
            },
            {
                "model": "Popularity baseline",
                "evaluation_users": len(pop_eval),
                "source_rows_used": len(rating_sample),
                "positive_interactions_used": len(filtered),
                "precision_at_10": pop_eval["precision_at_10"].mean(),
                "recall_at_10": pop_eval["recall_at_10"].mean(),
                "accuracy_hit_rate_at_10": pop_eval["hit_rate_at_10"].mean(),
            },
        ]
    )
    metrics.to_csv(os.path.join(OUT_DIR, "recommendation_evaluation_metrics.csv"), index=False)

    print("Stage 3 prototype pipeline completed.")
