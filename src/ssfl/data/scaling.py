"""Feature scaling (min-max, rank/quantile) and the Equation 19 (115 -> 23x5) reshape."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ssfl.config import NormalizationMode


@dataclass(frozen=True)
class MinMaxScaler:
    min_: np.ndarray  # (num_features,) float32
    max_: np.ndarray  # (num_features,) float32
    constant_mask: np.ndarray  # (num_features,) bool -- True where max == min

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.float32)
        non_constant = ~self.constant_mask
        scale = (self.max_ - self.min_)[non_constant]
        out[:, non_constant] = (x[:, non_constant] - self.min_[non_constant]) / scale
        return out  # constant features stay 0.0 (already zero-initialized)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"min": self.min_, "max": self.max_, "constant_mask": self.constant_mask}

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "MinMaxScaler":
        return cls(
            min_=arrays["min"].astype(np.float32),
            max_=arrays["max"].astype(np.float32),
            constant_mask=arrays["constant_mask"].astype(bool),
        )


@dataclass(frozen=True)
class QuantileScaler:
    """Per-feature rank transform: each value maps to its average rank in the fit set, in [0, 1].

    Min-max (and every other affine map) cannot separate gafgyt.tcp from gafgyt.udp: the five
    features that distinguish them span ~1.5e9 while the two classes differ by ~183 raw units, so
    after scaling the gap is 2 float32 ULPs no matter what offset and scale are chosen. Ranking is
    monotone but non-affine -- the gap becomes the number of fit samples lying between the two
    values, which for those features is thousands.

    This only helps if the fit matrix still holds the distinctions. It is why ``load_source_matrix``
    reads float64: under the old float32 load the rows were already identical here, and ranking
    identical values returns identical ranks, so this scaler was a no-op for the pair it exists to
    separate.

    Stored ragged (one sorted unique-value array per feature) as three flat arrays, because the
    per-feature unique counts range from a handful to tens of thousands.
    """

    values: np.ndarray  # (total,) float64 -- sorted unique fit values, features concatenated
    ranks: np.ndarray  # (total,) float32 -- matching average rank in [0, 1]
    offsets: np.ndarray  # (num_features + 1,) int64 -- feature j occupies [offsets[j], offsets[j+1])
    constant_mask: np.ndarray  # (num_features,) bool -- True where the feature has one value

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(x.shape, dtype=np.float32)
        for j in range(x.shape[1]):
            if self.constant_mask[j]:
                continue  # constant features stay 0.0, matching MinMaxScaler
            a, b = int(self.offsets[j]), int(self.offsets[j + 1])
            # np.interp clamps outside the fit range, so unseen extremes saturate at 0.0/1.0.
            out[:, j] = np.interp(x[:, j].astype(np.float64), self.values[a:b], self.ranks[a:b])
        return out

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "values": self.values,
            "ranks": self.ranks,
            "offsets": self.offsets,
            "constant_mask": self.constant_mask,
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "QuantileScaler":
        return cls(
            values=arrays["values"].astype(np.float64),
            ranks=arrays["ranks"].astype(np.float32),
            offsets=arrays["offsets"].astype(np.int64),
            constant_mask=arrays["constant_mask"].astype(bool),
        )


def fit_minmax_scaler(matrix: np.ndarray) -> MinMaxScaler:
    min_ = matrix.min(axis=0).astype(np.float32)
    max_ = matrix.max(axis=0).astype(np.float32)
    constant_mask = (max_ - min_) == 0
    return MinMaxScaler(min_=min_, max_=max_, constant_mask=constant_mask)


def fit_quantile_scaler(matrix: np.ndarray) -> QuantileScaler:
    n = len(matrix)
    values, ranks, offsets = [], [], [0]
    constant = []
    for j in range(matrix.shape[1]):
        unique, counts = np.unique(matrix[:, j].astype(np.float64), return_counts=True)
        # Average rank of each unique value (ties share a rank), normalised to [0, 1].
        upper = np.cumsum(counts)
        avg_rank = (upper - (counts - 1) / 2.0 - 1) / max(n - 1, 1)
        values.append(unique)
        ranks.append(avg_rank.astype(np.float32))
        offsets.append(offsets[-1] + len(unique))
        constant.append(len(unique) == 1)
    return QuantileScaler(
        values=np.concatenate(values),
        ranks=np.concatenate(ranks),
        offsets=np.asarray(offsets, dtype=np.int64),
        constant_mask=np.asarray(constant, dtype=bool),
    )


def fit_scaler(
    matrix: np.ndarray, mode: NormalizationMode = NormalizationMode.all_mini
) -> MinMaxScaler | QuantileScaler:
    if mode == NormalizationMode.quantile:
        return fit_quantile_scaler(matrix)
    return fit_minmax_scaler(matrix)


def reshape_eq19(x: np.ndarray, rows: int = 23, cols: int = 5) -> np.ndarray:
    """Equation 19: flat feature vector -> (rows, cols) matrix with ``M[r, c] = v[r + rows*c]``.

    Equivalent to ``v.reshape((rows, cols), order='F')`` for a single vector; implemented via a
    C-order reshape-then-transpose so it vectorizes correctly over any number of leading batch
    dimensions (``order='F'`` reshape does not compose with a leading batch axis).
    """
    if x.shape[-1] != rows * cols:
        raise ValueError(f"expected last dimension {rows * cols}, got {x.shape[-1]}")
    leading = x.shape[:-1]
    reshaped = x.reshape(*leading, cols, rows)
    axes = list(range(reshaped.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return reshaped.transpose(axes)
