"""
Data loading and episodic sampling for MSG* baseline experiments.

Implements N-way K-shot Q-query episode generation from TUDatasets
following the protocol in Snell et al. (2017) Prototypical Networks.
"""

import torch
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


class TUGraphDataset:
    """
    Load a TU graph dataset with train/test split.

    Attributes:
        name: Dataset name (e.g., 'MUTAG', 'PROTEINS').
        train_graphs: List of (Data, label) for training.
        test_graphs: List of (Data, label) for testing.
        num_classes: Number of classes.
        num_features: Input feature dimension.
    """

    def __init__(
        self,
        name: str,
        tu_name: str,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.name = name
        self.tu_name = tu_name

        # Download/load dataset
        full_data = TUDataset(root="./data/TU", name=tu_name)

        # Group graphs by label
        class_indices: Dict[int, List[int]] = {}
        for idx, data in enumerate(full_data):
            label = int(data.y.item())
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        self.num_classes = len(class_indices)
        self.num_features = full_data.num_features

        # Per-class stratified split
        rng = random.Random(seed)
        self.train_graphs: List[Tuple[torch.Tensor, int]] = []  # (graph_data, label)
        self.test_graphs: List[Tuple[torch.Tensor, int]] = []

        for label, indices in sorted(class_indices.items()):
            rng.shuffle(indices)
            n_train = max(1, int(len(indices) * train_ratio))
            for idx in indices[:n_train]:
                self.train_graphs.append((full_data[idx], label))
            for idx in indices[n_train:]:
                self.test_graphs.append((full_data[idx], label))

        print(
            f"[{name}] Loaded {len(self.train_graphs)} train / "
            f"{len(self.test_graphs)} test graphs | "
            f"{self.num_classes} classes | {self.num_features} features"
        )


class EpisodeSampler:
    """
    Sample N-way K-shot Q-query episodes for few-shot learning.

    Each episode consists of:
      - support_set: N classes × K graphs (used to compute prototypes)
      - query_set:   N classes × Q graphs (used to evaluate classification)

    Protocol follows Snell et al. (2017) ProtoNet.
    """

    def __init__(
        self,
        dataset: TUGraphDataset,
        n_way: int,
        k_shot: int,
        q_query: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dataset: TUGraphDataset with train/test splits.
            n_way: Number of classes per episode (N).
            k_shot: Support examples per class (K).
            q_query: Query examples per class (Q).
            seed: Random seed for reproducibility.
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.rng = random.Random(seed)

        # Organize training graphs by class
        self.class_graphs: Dict[int, List] = {}
        for graph, label in dataset.train_graphs:
            if label not in self.class_graphs:
                self.class_graphs[label] = []
            self.class_graphs[label].append(graph)

        # Validate we have enough classes and graphs per class
        available_classes = sorted(self.class_graphs.keys())
        assert len(available_classes) >= n_way, (
            f"Need {n_way} classes but only {len(available_classes)} available"
        )
        for c in available_classes[:n_way]:
            assert len(self.class_graphs[c]) >= k_shot + q_query, (
                f"Class {c} has only {len(self.class_graphs[c])} graphs, "
                f"need K+Q={k_shot + q_query}"
            )

    def sample_episode(self, mode: str = "train") -> dict:
        """
        Sample a single N-way K-shot episode.

        Args:
            mode: 'train' or 'test'.

        Returns:
            Dict with keys:
              - 'support': list of N*K Data objects
              - 'support_labels': list of N*K ints
              - 'query': list of N*Q Data objects
              - 'query_labels': list of N*Q ints
              - 'way_labels': list of N unique class labels used
        """
        if mode == "test":
            # Use test set for evaluation episodes
            class_graphs_test: Dict[int, List] = {}
            for graph, label in self.dataset.test_graphs:
                if label not in class_graphs_test:
                    class_graphs_test[label] = []
                class_graphs_test[label].append(graph)
            pool = class_graphs_test
        else:
            pool = self.class_graphs

        # Select N classes uniformly at random
        available = sorted(pool.keys())
        chosen_classes = self.rng.sample(available, self.n_way)

        support_graphs = []
        support_labels = []
        query_graphs = []
        query_labels = []

        for cls_idx, cls_label in enumerate(chosen_classes):
            graphs = pool[cls_label]
            self.rng.shuffle(graphs)

            # Take first K as support, next Q as query
            supp = graphs[: self.k_shot]
            quer = graphs[self.k_shot : self.k_shot + self.q_query]

            support_graphs.extend(supp)
            support_labels.extend([cls_idx] * self.k_shot)
            query_graphs.extend(quer)
            query_labels.extend([cls_idx] * self.q_query)

        return {
            "support": support_graphs,
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query": query_graphs,
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
            "way_labels": chosen_classes,
        }

    def sample_batch(
        self, n_episodes: int, mode: str = "train"
    ) -> List[dict]:
        """Sample multiple independent episodes."""
        return [self.sample_episode(mode=mode) for _ in range(n_episodes)]


def collate_graphs(graph_list: List):
    """
    Batch a list of PyG Data objects into a single batched graph.
    Uses standard PyG batching (disjoint union + batch vector).
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(graph_list)


def batch_episode_data(
    episode: dict, device: str = "cpu"
) -> Tuple:
    """
    Collate an episode's support/query sets into batched PyG graphs.

    Returns:
        support_batch, support_labels, query_batch, query_labels
    """
    support_batch = collate_graphs(episode["support"]).to(device)
    query_batch = collate_graphs(episode["query"]).to(device)
    return (
        support_batch,
        episode["support_labels"].to(device),
        query_batch,
        episode["query_labels"].to(device),
    )
