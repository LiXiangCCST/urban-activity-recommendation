"""Placeholder graph encoder for future GNN models."""

from __future__ import annotations

import networkx as nx
import numpy as np


class GNNEncoder:
    """Minimal encoder interface.

    TODO: Replace this with a message-passing neural network over area/resource
    nodes. The expected output is a dense embedding per area node.
    """

    def __init__(self, embedding_dim: int = 8) -> None:
        self.embedding_dim = embedding_dim

    def encode(self, graph: nx.Graph) -> dict[str, np.ndarray]:
        embeddings: dict[str, np.ndarray] = {}
        for node_id, attrs in graph.nodes(data=True):
            degree = graph.degree[node_id]
            node_type = 1.0 if attrs.get("node_type") == "area" else 0.0
            vector = np.zeros(self.embedding_dim, dtype=float)
            vector[0] = node_type
            vector[1] = degree
            if self.embedding_dim > 2:
                vector[2] = float(attrs.get("risk_score", attrs.get("quality", 0.0)))
            embeddings[node_id] = vector
        return embeddings
