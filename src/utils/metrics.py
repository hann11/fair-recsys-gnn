from typing import List

import numpy as np
import pandas as pd


def dcg_at_k(r: List[int], k: int) -> float:
    r_k = np.asfarray(r)[:k]
    return np.sum(r_k / np.log2(np.arange(2, r_k.size + 2)))


def precision_at_k(r: List[int], k: int) -> float:
    r_k = np.asarray(r)[:k]
    return np.mean(r_k)


def ndcg_at_k(r: List[int], k: int) -> float:
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    return dcg_at_k(r, k) / dcg_max


def recall_at_k(r: List[int], k: int, n_pos_items: int) -> float:
    r_k = np.asfarray(r)[:k]
    return np.sum(r_k) / n_pos_items


def hit_at_k(r: List[int], k: int) -> float:
    r_k = np.array(r)[:k]
    if np.sum(r_k) > 0:
        return 1
    else:
        return 0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall > 0:
        return (2.0 * precision * recall) / (precision + recall)
    else:
        return 0


def diversity(item_ids: List[int], item_embed: pd.DataFrame) -> float:
    n_items = len(item_ids)

    if n_items == 1:
        return 1

    div = 0.0  # type: ignore

    df_embed = item_embed[item_embed["item_id"].isin(item_ids)]

    np_embed = np.array(df_embed.drop(columns="item_id"))

    for i in range(n_items):
        for j in range(n_items):
            if i != j:
                div += np.linalg.norm(np_embed[i] - np_embed[j])  # type: ignore

    return div / n_items * (n_items - 1)


def novelty(item_ids: List[int], item_counts: pd.DataFrame) -> float:
    n_items = len(item_ids)

    df_novelty = item_counts[item_counts["item_id"].isin(item_ids)]

    nov = np.sum(df_novelty["novelty"])

    return nov / n_items
