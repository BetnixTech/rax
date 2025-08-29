# rax.py
"""
Rax — a small Learning-to-Rank toolkit in JAX.

Improvements over the original:
- Prefer jax.numpy for JAX-level operations inside functions used in jitted code.
- Cleaner MLP init / forward behavior and explicit expectation for scalar score output.
- Slightly safer metric computations with shape guards.
- Clearer docstrings and typing for experimentation.
"""

from typing import Any, Callable, Dict, List, Tuple
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclasses.dataclass
class GroupedDataset:
    """
    Holds queries with variable-length lists of documents.

    Example group:
      {"features": np.array([...docs x D]), "labels": np.array([...docs])}
    """
    groups: List[Dict[str, np.ndarray]]

    def group_sizes(self) -> List[int]:
        return [g["features"].shape[0] for g in self.groups]

    def as_padded(self, pad_to: int = None) -> Dict[str, np.ndarray]:
        """Return padded arrays (batch, max_docs, feat_dim) for features and (batch, max_docs) for labels/mask."""
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


class MLP:
    """Simple functional MLP used as ranking model. The final layer should produce a scalar score per example."""

    def __init__(self, layer_sizes: List[int], key: jax.random.KeyArray):
        """
        layer_sizes: e.g. [feature_dim, hidden1, ..., 1] - final dim should be 1 for scalar score.
        """
        assert len(layer_sizes) >= 2, "Need at least input and output sizes"
        self.layer_sizes = layer_sizes
        self.params = self._init_params(layer_sizes, key)

    @staticmethod
    def _init_params(sizes: List[int], key: jax.random.KeyArray) -> Dict[str, Dict[str, jnp.ndarray]]:
        params = {}
        keys = jax.random.split(key, len(sizes) - 1)
        for i in range(len(sizes) - 1):
            k = keys[i]
            w = jax.random.normal(k, (sizes[i], sizes[i + 1])) * (1.0 / math.sqrt(max(1, sizes[i])))
            b = jnp.zeros((sizes[i + 1],), dtype=jnp.float32)
            params[f"layer_{i}"] = {"w": w, "b": b}
        return params

    @staticmethod
    def forward(params: Dict[str, Dict[str, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (..., feature_dim)
        Returns: (..., out_dim). For ranking use-case we expect out_dim == 1 so callers can squeeze to (...) if desired.
        """
        h = x
        n_layers = len(params)
        for i in range(n_layers):
            p = params[f"layer_{i}"]
            h = jnp.dot(h, p["w"]) + p["b"]
            if i < n_layers - 1:
                h = jax.nn.relu(h)
        return h  # caller can squeeze if final dim == 1


def pairwise_hinge_loss(scores: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray, margin: float = 1.0) -> jnp.ndarray:
    """
    Pairwise hinge loss.
      scores: (B, D)
      labels: (B, D)
      mask:   (B, D) (1.0 valid, 0.0 pad)
    """
    s_i = jnp.expand_dims(scores, 2)  # (B, D, 1)
    s_j = jnp.expand_dims(scores, 1)  # (B, 1, D)
    l_i = jnp.expand_dims(labels, 2)
    l_j = jnp.expand_dims(labels, 1)
    mask_i = jnp.expand_dims(mask, 2)
    mask_j = jnp.expand_dims(mask, 1)
    valid_pair = (l_i > l_j) & (mask_i == 1.0) & (mask_j == 1.0)
    margins = jnp.maximum(0.0, margin - (s_i - s_j))
    loss = jnp.where(valid_pair, margins, 0.0)
    pair_count = jnp.maximum(1.0, jnp.sum(valid_pair, axis=(1, 2)))
    per_query_loss = jnp.sum(loss, axis=(1, 2)) / pair_count
    return jnp.mean(per_query_loss)


def listwise_softmax_ce(scores: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    labels_masked = labels * mask
    denom = jnp.sum(labels_masked, axis=1, keepdims=True) + eps
    target_dist = labels_masked / denom
    pred_logprob = jax.nn.log_softmax(scores, axis=1)
    loss = -jnp.sum(target_dist * pred_logprob * mask, axis=1)
    return jnp.mean(loss)


def dcg_at_k(rels: jnp.ndarray, k: int, mask: jnp.ndarray) -> jnp.ndarray:
    k = min(k, rels.shape[1])
    ranks = jnp.arange(1, k + 1)
    discounts = 1.0 / jnp.log2(ranks + 1.0)
    sorted_rels = jnp.sort(rels * mask, axis=1)[:, ::-1][:, :k]
    return jnp.sum(sorted_rels * discounts, axis=1)


def ndcg_at_k(scores: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray, k: int = 10) -> float:
    """
    Compute NDCG@k for a batch.
    """
    # indices sorted by score descending
    idx = jnp.argsort(scores, axis=1)[:, ::-1]
    batch_idx = jnp.expand_dims(jnp.arange(scores.shape[0]), 1)
    ranked_labels = labels[batch_idx, idx]
    ranked_mask = mask[batch_idx, idx]
    dcg = dcg_at_k(ranked_labels, k, ranked_mask)

    ideal_idx = jnp.argsort(labels * mask, axis=1)[:, ::-1]
    ideal_labels = labels[batch_idx, ideal_idx]
    ideal_mask = mask[batch_idx, ideal_idx]
    idcg = dcg_at_k(ideal_labels, k, ideal_mask)
    ndcg = jnp.where(idcg > 0, dcg / idcg, 0.0)
    return float(jnp.mean(ndcg))


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
    batch: {'features': (B, D, F), 'labels': (B, D), 'mask': (B, D)}
    """

    def loss_and_grads(params, batch):
        feats = batch["features"]  # (B, D, F)
        B, D, F = feats.shape
        flat_feats = feats.reshape((B * D, F))
        raw_scores = model_forward(params, flat_feats)  # (B*D, 1) or (B*D,)
        # ensure shape (B*D,)
        if raw_scores.ndim == 2 and raw_scores.shape[-1] == 1:
            raw_scores = raw_scores.reshape((B * D,))
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


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    g1 = {"features": np.random.randn(3, 16).astype(np.float32), "labels": np.array([3.0, 2.0, 1.0], dtype=np.float32)}
    g2 = {"features": np.random.randn(4, 16).astype(np.float32), "labels": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)}
    ds = GroupedDataset([g1, g2])
    batch = ds.as_padded()

    mlp = MLP([16, 32, 16, 1], key)
    optimizer = optax.adam(1e-3)
    state = create_train_state(mlp.params, optimizer)

    update_fn = make_update_fn(MLP.forward, pairwise_hinge_loss, optimizer)
    jax_batch = {k: jnp.array(v) for k, v in batch.items()}
    state, metrics = update_fn(state, jax_batch)
    print("Demo update metrics:", metrics)

    flat_feats = jax_batch["features"].reshape((2 * jax_batch["features"].shape[1], -1))
    raw_scores = MLP.forward(state.params, flat_feats)
    if raw_scores.ndim == 2 and raw_scores.shape[-1] == 1:
        raw_scores = raw_scores.reshape((2 * jax_batch["features"].shape[1],))
    raw_scores = raw_scores.reshape((2, -1))
    print("NDCG@3:", ndcg_at_k(raw_scores, jax_batch["labels"], jax_batch["mask"], k=3))
