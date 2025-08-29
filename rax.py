# rax.py
"""
Rax — a tiny Learning-to-Rank library in JAX.

Features:
- Simple dataset helper for query/doc groups
- Small MLP ranking model wrapper
- Pairwise hinge (RankNet-style) and softmax-listwise loss
- NDCG metric
- Training loop utilities (jit'd step, basic batching)

Dependencies:
- jax
- optax
- numpy

This is intentionally minimal and meant to be adapted for experiments.
"""

from typing import Any, Callable, Dict, Iterable, List, Tuple
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax


# ---------- Data helpers ----------
@dataclasses.dataclass
class GroupedDataset:
    """
    Holds queries with variable-length lists of documents.

    Example:
      data = [
        {"features": np.array([...docs x D]), "labels": np.array([...docs])},
        ...
      ]
    """
    groups: List[Dict[str, np.ndarray]]

    def group_sizes(self) -> List[int]:
        return [g["features"].shape[0] for g in self.groups]

    def as_padded(self, pad_to: int = None) -> Dict[str, np.ndarray]:
        """Return (batch_size, max_docs, feat_dim) padded arrays for features and labels."""
        batch = len(self.groups)
        feat_dim = self.groups[0]["features"].shape[1]
        max_docs = pad_to if pad_to is not None else max(self.group_sizes())
        feats = np.zeros((batch, max_docs, feat_dim), dtype=self.groups[0]["features"].dtype)
        labels = np.full((batch, max_docs), -1.0, dtype=self.groups[0]["labels"].dtype)  # -1 for padding
        mask = np.zeros((batch, max_docs), dtype=np.float32)
        for i, g in enumerate(self.groups):
            n = g["features"].shape[0]
            feats[i, :n] = g["features"]
            labels[i, :n] = g["labels"]
            mask[i, :n] = 1.0
        return {"features": feats, "labels": labels, "mask": mask}


# ---------- Model ----------
class MLP:
    """Simple MLP ranking model built with pure JAX (functional)."""

    def __init__(self, layer_sizes: List[int], key: jax.random.KeyArray):
        self.layer_sizes = layer_sizes
        self.params = self._init_params(layer_sizes, key)

    @staticmethod
    def _init_params(sizes: List[int], key: jax.random.KeyArray) -> Dict[str, Dict[str, jnp.ndarray]]:
        params = {}
        keys = jax.random.split(key, len(sizes) - 1)
        for i in range(len(sizes) - 1):
            k1, k2 = jax.random.split(keys[i])
            w = jax.random.normal(k1, (sizes[i], sizes[i + 1])) * (1.0 / math.sqrt(sizes[i]))
            b = jnp.zeros((sizes[i + 1],))
            params[f"layer_{i}"] = {"w": w, "b": b}
        return params

    @staticmethod
    def forward(params: Dict[str, Dict[str, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
        h = x
        n_layers = len(params)
        for i in range(n_layers):
            p = params[f"layer_{i}"]
            h = jnp.dot(h, p["w"]) + p["b"]
            if i < n_layers - 1:
                h = jax.nn.relu(h)
        # final scalar score per example
        return h.squeeze(-1) if h.ndim == 2 and h.shape[-1] == 1 else h


# ---------- Losses ----------
def pairwise_hinge_loss(scores: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray, margin: float = 1.0) -> jnp.ndarray:
    """
    Pairwise hinge loss. Expects shapes:
      scores: (batch, docs)
      labels: (batch, docs)  (higher = better)
      mask:   (batch, docs)  (1.0 valid, 0.0 pad)
    For each pair (i,j) where label_i > label_j, loss = max(0, margin - (s_i - s_j))
    """
    # Expand dims for pairwise comparisons
    s_i = jnp.expand_dims(scores, 2)  # (B, D, 1)
    s_j = jnp.expand_dims(scores, 1)  # (B, 1, D)
    l_i = jnp.expand_dims(labels, 2)
    l_j = jnp.expand_dims(labels, 1)
    mask_i = jnp.expand_dims(mask, 2)
    mask_j = jnp.expand_dims(mask, 1)
    valid_pair = (l_i > l_j) & (mask_i == 1.0) & (mask_j == 1.0)
    margins = jnp.maximum(0.0, margin - (s_i - s_j))
    loss = jnp.where(valid_pair, margins, 0.0)
    # Normalize by number of valid pairs per batch
    pair_count = jnp.maximum(1.0, jnp.sum(valid_pair, axis=(1, 2)))
    per_query_loss = jnp.sum(loss, axis=(1, 2)) / pair_count
    return jnp.mean(per_query_loss)


def listwise_softmax_ce(scores: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """
    Listwise softmax cross-entropy: treat labels as targets (convert to probabilities).
      - convert labels to distribution (e.g., soft labels)
      - compute cross-entropy between label distribution and softmax(scores)
    """
    # mask out pads: set label for pads to 0 and exclude from normalization
    labels_masked = labels * mask
    denom = jnp.sum(labels_masked, axis=1, keepdims=True) + eps
    target_dist = labels_masked / denom  # (B, D)
    pred_logprob = jax.nn.log_softmax(scores, axis=1)
    loss = -jnp.sum(target_dist * pred_logprob * mask, axis=1)
    return jnp.mean(loss)


# ---------- Metrics ----------
def dcg_at_k(rels: jnp.ndarray, k: int, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Compute DCG@k for a batch: rels shape (B, D), mask shape (B, D)
    """
    k = min(k, rels.shape[1])
    ranks = jnp.arange(1, k + 1)
    discounts = 1.0 / jnp.log2(ranks + 1.0)
    # sort rels descending
    sorted_rels = jnp.sort(rels * mask, axis=1)[:, ::-1][:, :k]
    return jnp.sum(sorted_rels * discounts, axis=1)


def ndcg_at_k(scores: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray, k: int = 10) -> float:
    """
    Compute NDCG@k:
      - Use predicted scores to sort, compute DCG
      - Compute ideal DCG from labels
    """
    # Get indices sorted by scores descending
    idx = jnp.argsort(scores, axis=1)[:, ::-1]
    batch_indices = jnp.expand_dims(jnp.arange(scores.shape[0]), 1)
    # gather labels by predicted ranking
    ranked_labels = labels[batch_indices, idx]
    ranked_mask = mask[batch_indices, idx]
    dcg = dcg_at_k(ranked_labels, k, ranked_mask)
    # ideal dcg (sorted by labels)
    ideal_idx = jnp.argsort(labels * mask, axis=1)[:, ::-1]
    ideal_labels = labels[batch_indices, ideal_idx]
    ideal_mask = mask[batch_indices, ideal_idx]
    idcg = dcg_at_k(ideal_labels, k, ideal_mask)
    ndcg = jnp.where(idcg > 0, dcg / idcg, 0.0)
    return float(jnp.mean(ndcg))


# ---------- Training utilities ----------
@dataclasses.dataclass
class TrainState:
    params: Dict[str, Dict[str, jnp.ndarray]]
    opt_state: optax.OptState
    step: int = 0


def create_train_state(params: Dict[str, Any], optimizer: optax.GradientTransformation) -> TrainState:
    opt_state = optimizer.init(params)
    return TrainState(params=params, opt_state=opt_state, step=0)


def make_update_fn(model_forward: Callable, loss_fn: Callable, optimizer: optax.GradientTransformation):
    """
    Returns a jit'd update function:
      (state, batch) -> new_state, metrics
    where batch is dict with 'features', 'labels', 'mask'
    """

    def loss_and_grads(params, batch):
        feats = batch["features"]  # (B, D, F)
        # flatten docs into examples for scoring
        B, D, F = feats.shape
        flat_feats = feats.reshape((B * D, F))
        raw_scores = model_forward(params, flat_feats)  # (B*D,) or (B*D,1)
        scores = raw_scores.reshape((B, D))
        loss = loss_fn(scores, batch["labels"], batch["mask"])
        return loss, scores

    @jax.jit
    def update(state: TrainState, batch: Dict[str, jnp.ndarray]):
        (loss_val, scores), grads = jax.value_and_grad(loss_and_grads, has_aux=True)(state.params, batch)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_state = TrainState(params=new_params, opt_state=new_opt_state, step=state.step + 1)
        metrics = {"loss": float(loss_val)}
        return new_state, metrics

    return update


# ---------- Example usage ----------
if __name__ == "__main__":
    # tiny demo to sanity-check API
    key = jax.random.PRNGKey(0)
    # Create synthetic group dataset: 2 queries, varying docs
    g1 = {"features": np.random.randn(3, 16).astype(np.float32), "labels": np.array([3.0, 2.0, 1.0], dtype=np.float32)}
    g2 = {"features": np.random.randn(4, 16).astype(np.float32), "labels": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)}
    ds = GroupedDataset([g1, g2])
    batch = ds.as_padded()

    # build model that maps feature_dim -> 1 score
    mlp = MLP([16, 32, 16, 1], key)
    optimizer = optax.adam(1e-3)
    state = create_train_state(mlp.params, optimizer)

    # Define loss (pairwise hinge) and update function
    update_fn = make_update_fn(MLP.forward, pairwise_hinge_loss, optimizer)

    # single update step
    jax_batch = {k: jnp.array(v) for k, v in batch.items()}
    state, metrics = update_fn(state, jax_batch)
    print("Demo update metrics:", metrics)

    # compute ndcg
    # score with fresh params
    flat_feats = jax_batch["features"].reshape((2 * jax_batch["features"].shape[1], -1))
    raw_scores = MLP.forward(state.params, flat_feats).reshape((2, -1))
    print("NDCG@3:", ndcg_at_k(raw_scores, jax_batch["labels"], jax_batch["mask"], k=3))
