"""
VectorForge — In-Memory Vector Store

Central storage layer backed by NumPy arrays.
All indexes reference this store rather than maintaining their own copies.

Layout:
    vectors     : np.ndarray  (N, D) float32 — normalized embeddings
    ids         : np.ndarray  (N,)           — string identifiers
    metadata    : dict[str, dict]            — optional per-vector metadata
    active_mask : np.ndarray  (N,) bool      — True = active, False = deleted
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.core.distance import normalize
from app.core.exceptions import (
    DimensionMismatchError,
    DuplicateVectorError,
    VectorNotFoundError,
)


class VectorStore:
    """
    In-memory vector storage with logical deletion.

    Vectors are normalized on insertion so that cosine similarity
    reduces to a simple dot product at query time.
    """

    def __init__(self, dimension: int):
        self.dimension: int = dimension
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.ids: np.ndarray = np.empty(0, dtype=object)
        self.active_mask: np.ndarray = np.empty(0, dtype=bool)
        self.metadata: dict[str, dict] = {}

        # Fast ID → row-index lookup
        self._id_to_idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------
    def insert(
        self,
        vector_id: str,
        vector: np.ndarray | list,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Insert a vector. Normalizes it automatically."""
        vector_id = str(vector_id)

        if vector_id in self._id_to_idx:
            raise DuplicateVectorError(vector_id)

        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != self.dimension:
            raise DimensionMismatchError(self.dimension, vec.shape[-1] if vec.size else 0)

        # Normalize once at ingestion
        vec = normalize(vec)

        idx = len(self.ids)
        self.vectors = np.vstack([self.vectors, vec.reshape(1, -1)]) if idx > 0 else vec.reshape(1, -1)
        self.ids = np.append(self.ids, vector_id)
        self.active_mask = np.append(self.active_mask, True)
        self._id_to_idx[vector_id] = idx

        if meta:
            self.metadata[vector_id] = meta

    # ------------------------------------------------------------------
    # Bulk load (for dataset generation / startup)
    # ------------------------------------------------------------------
    def bulk_insert(
        self,
        ids: np.ndarray | list,
        vectors: np.ndarray,
        metadata_list: list[dict] | None = None,
    ) -> int:
        """
        Insert many vectors at once — much faster than repeated insert().

        Parameters
        ----------
        ids      : array-like of string IDs, length N
        vectors  : np.ndarray shape (N, D), will be normalized
        metadata_list : optional list of dicts, length N
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise DimensionMismatchError(
                self.dimension,
                vectors.shape[1] if vectors.ndim == 2 else -1,
            )

        ids_arr = np.asarray(ids, dtype=object)

        # Check for duplicates
        for vid in ids_arr:
            vid = str(vid)
            if vid in self._id_to_idx:
                raise DuplicateVectorError(vid)

        # Normalize batch
        normed = normalize(vectors)

        start = len(self.ids)
        if start == 0:
            self.vectors = normed
        else:
            self.vectors = np.vstack([self.vectors, normed])

        self.ids = np.concatenate([self.ids, ids_arr])
        self.active_mask = np.concatenate(
            [self.active_mask, np.ones(len(ids_arr), dtype=bool)]
        )

        for i, vid in enumerate(ids_arr):
            vid = str(vid)
            self._id_to_idx[vid] = start + i
            if metadata_list and i < len(metadata_list) and metadata_list[i]:
                self.metadata[vid] = metadata_list[i]
        
        return len(ids_arr)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def get(self, vector_id: str) -> dict:
        """Return vector data by ID. Raises VectorNotFoundError if missing."""
        vector_id = str(vector_id)
        if vector_id not in self._id_to_idx:
            raise VectorNotFoundError(vector_id)

        idx = self._id_to_idx[vector_id]
        return {
            "id": vector_id,
            "vector": self.vectors[idx],
            "active": bool(self.active_mask[idx]),
            "metadata": self.metadata.get(vector_id, None),
        }

    # ------------------------------------------------------------------
    # Delete (logical)
    # ------------------------------------------------------------------
    def delete(self, vector_id: str) -> None:
        """Mark a vector as deleted (logical deletion)."""
        vector_id = str(vector_id)
        if vector_id not in self._id_to_idx:
            raise VectorNotFoundError(vector_id)

        idx = self._id_to_idx[vector_id]
        self.active_mask[idx] = False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_active_vectors(self) -> np.ndarray:
        """Return (M, D) array of active vectors only."""
        return self.vectors[self.active_mask]

    def get_active_ids(self) -> np.ndarray:
        """Return IDs of active vectors only."""
        return self.ids[self.active_mask]

    def get_all_vectors(self) -> np.ndarray:
        """Return all vectors (including deleted)."""
        return self.vectors

    def get_all_ids(self) -> np.ndarray:
        """Return all IDs (including deleted)."""
        return self.ids

    def count(self) -> int:
        """Return the number of active (non-deleted) vectors."""
        return int(np.sum(self.active_mask))

    def total_count(self) -> int:
        """Return total number of vectors including deleted."""
        return len(self.ids)

    def is_empty(self) -> bool:
        return self.total_count() == 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        """Save vectors, IDs, and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "vectors.npy"), self.vectors)
        np.save(os.path.join(directory, "ids.npy"), self.ids)
        np.save(os.path.join(directory, "active_mask.npy"), self.active_mask)
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, directory: str) -> None:
        """Load vectors, IDs, and metadata from disk."""
        self.vectors = np.load(
            os.path.join(directory, "vectors.npy"), allow_pickle=False
        ).astype(np.float32)
        self.ids = np.load(
            os.path.join(directory, "ids.npy"), allow_pickle=True
        )
        active_path = os.path.join(directory, "active_mask.npy")
        if os.path.exists(active_path):
            self.active_mask = np.load(active_path, allow_pickle=False).astype(bool)
        else:
            self.active_mask = np.ones(len(self.ids), dtype=bool)

        meta_path = os.path.join(directory, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Rebuild index
        self._id_to_idx = {str(vid): i for i, vid in enumerate(self.ids)}
        self.dimension = self.vectors.shape[1] if len(self.vectors) > 0 else self.dimension
