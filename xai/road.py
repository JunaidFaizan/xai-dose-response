"""Exact ROAD: Noisy Linear Imputation with MoRF and LeRF removal (amendment A3, SD3).

Rong, Leemann, Borisov, Kasneci and Kasneci, "A Consistent and Efficient
Evaluation Strategy for Attribution Methods", ICML 2022,
https://arxiv.org/abs/2202.00449.

Ported from the reference code https://github.com/tleemann/road_evaluation
(file road/imputations.py, class NoisyLinearImputer, commit
e6b9987ffbd79654125af2f581b5dae70e831b09): the neighbour weights, the sparse
linear system (each removed pixel equals the weighted mean of its in-image
neighbours, solved jointly per image and channel with scipy's spsolve) and the
Gaussian noise added to the imputed pixels only, without clipping. Changes:
the system matrix is assembled as a COO matrix instead of a LIL matrix (the
same entries), the removed pixels are passed as a boolean mask (True =
impute) instead of the reference keep-mask (0 = impute), and the noise comes
from an explicit CPU generator. tests/test_road.py checks the port against
outputs of the reference code (tests/fixtures/road_nli_fixture.npz).

The removal grid, orders and area follow amendment A3
(docs/PREDICTIONS_amendments.md): q = share of pixels removed, both orders
remove int(n * q) pixels, and the area over q in [0, 0.9] is divided by 0.9.

Licence of the reference code, reproduced as it requires:

    MIT License

    Copyright (c) 2022 Tobias Leemann

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import zlib

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# Shares of pixels removed; q = 0 is the unmodified image (A3: the reference
# demo grid 0.1-0.5, 0.7, 0.9 with q = 0 added, 8 curve points).
ROAD_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
ROAD_NOISE = 0.01  # reference default, absolute, on images in [0, 1]
# ((row offset, column offset), weight), in the reference order: 1/12 for the
# diagonal neighbours, 1/6 for the edge neighbours.
NEIGHBOUR_WEIGHTS = [((1, 1), 1 / 12), ((0, 1), 1 / 6), ((-1, 1), 1 / 12), ((1, -1), 1 / 12),
                     ((0, -1), 1 / 6), ((-1, -1), 1 / 12), ((1, 0), 1 / 6), ((-1, 0), 1 / 6)]


def setup_sparse_system(removed, image, shape):
    """System A x = b for the removed pixels (reference setup_sparse_system).

    removed: (H*W,) bool, True = imputed. image: (C, H*W) values.
    Returns the (k, k) CSC matrix and the (k, C) right-hand side. Row i:
    -(sum of in-image neighbour weights) x_i + sum over removed neighbours
    w_j x_j = -sum over known neighbours w_j v_j.
    """
    height, width = shape
    indices = np.flatnonzero(removed)
    k = len(indices)
    variable = np.zeros(len(removed), dtype=np.int64)
    variable[indices] = np.arange(k)
    b = np.zeros((k, image.shape[0]))
    sum_neighbors = np.ones(k)
    rows, cols, data = [], [], []
    for (d_row, d_col), weight in NEIGHBOUR_WEIGHTS:
        row, col = indices // width + d_row, indices % width + d_col
        inside = (row >= 0) & (col >= 0) & (row < height) & (col < width)
        ids, coords = np.flatnonzero(inside), (row * width + col)[inside]
        known = ~removed[coords]
        b[ids[known], :] -= weight * image[:, coords[known]].T
        rows.append(ids[~known])
        cols.append(variable[coords[~known]])
        data.append(np.full(int((~known).sum()), weight))
        sum_neighbors[~inside] -= weight  # neighbours outside the image are dropped
    rows.append(np.arange(k))
    cols.append(np.arange(k))
    data.append(-sum_neighbors)
    matrix = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(k, k))
    return matrix.tocsc(), b


def noisy_linear_impute(image, removed, noise=ROAD_NOISE, generator=None):
    """Noisy Linear Imputation of the removed pixels of one image.

    image: (C, H, W) array or tensor with values in [0, 1].
    removed: (H, W) bool, True = removed and imputed.
    noise: standard deviation of the Gaussian noise added to the imputed
        pixels only (0 for none). generator: CPU torch.Generator for the noise.
    Returns a (C, H, W) float32 CPU tensor (a copy); not clipped.
    """
    image = torch.as_tensor(image).detach().cpu().to(torch.float32)
    channels, height, width = image.shape
    flat = image.reshape(channels, -1).clone()
    removed = np.asarray(removed, dtype=bool).reshape(-1)
    indices = np.flatnonzero(removed)
    if len(indices) == 0:
        return flat.reshape(channels, height, width)
    matrix, b = setup_sparse_system(removed, flat.numpy(), (height, width))
    # spsolve returns a 1-D vector for a single channel, so reshape to (k, C).
    solution = np.asarray(spsolve(matrix, b)).reshape(len(indices), channels)
    values = torch.tensor(solution, dtype=torch.float32).t()
    if noise:
        values = values + noise * torch.randn((channels, len(indices)), generator=generator)
    flat[:, torch.from_numpy(indices)] = values
    return flat.reshape(channels, height, width)


def removed_pixels(order, q, metric):
    """Flat indices removed at share q. order: pixels from most to least
    important. MoRF removes the int(n q) highest ranked, LeRF the int(n q)
    lowest ranked."""
    n = len(order)
    k = int(n * q)
    if metric == "road_morf":
        return order[:k]
    if metric == "road_lerf":
        return order[n - k:]
    raise ValueError(f"unknown ROAD metric {metric}")


def road_auc(curve):
    """Area under the probability curve over ROAD_GRID (trapezoid rule on the
    uneven grid), divided by 0.9 so a constant curve c scores c."""
    return float(np.trapezoid(curve, x=ROAD_GRID) / ROAD_GRID[-1])


def road_noise_seed(name, index):
    """Seed of the CPU noise generator for one model and image."""
    return zlib.crc32(f"{name}:{index}".encode())
