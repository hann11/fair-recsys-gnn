import numpy as np
from loguru import logger
from scipy import sparse


def get_top_k_scored_items(scores, top_k, sort_top_k=False):
    """Extract top K items from a matrix of scores for each user-item pair, optionally sort results per user.

    Args:
        scores (numpy.ndarray): Score matrix (users x items).
        top_k (int): Number of top items to recommend.
        sort_top_k (bool): Flag to sort top k results.

    Returns:
        numpy.ndarray, numpy.ndarray:
        - Indices into score matrix for each user's top items.
        - Scores corresponding to top items.

    """

    # ensure we're working with a dense ndarray
    if isinstance(scores, sparse.spmatrix):
        scores = scores.todense()

    if scores.shape[1] < top_k:
        logger.warning(
            "Number of items is less than top_k, limiting top_k to number of items"
        )
    k = min(top_k, scores.shape[1])

    test_user_idx = np.arange(scores.shape[0])[:, None]

    # get top K items and scores
    # this determines the un-ordered top-k item indices for each user
    top_items = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = scores[test_user_idx, top_items]

    if sort_top_k:
        sort_ind = np.argsort(-top_scores)
        top_items = top_items[test_user_idx, sort_ind]
        top_scores = top_scores[test_user_idx, sort_ind]

    return np.array(top_items), np.array(top_scores)
