import itertools
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist, squareform

if TYPE_CHECKING:
    from ._utilities import FloatArray

R = (1 + np.sqrt(5)) / 2
N_VERTICES_ICOSAHEDRON = 12


def normalize_to_unit_length(x: FloatArray) -> FloatArray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def find_unique_pairs(x: npt.NDArray[np.bool_]) -> npt.NDArray[np.integer]:
    return np.unique(np.sort(np.stack(np.nonzero(x), axis=-1), axis=-1), axis=0)


def subdivide_icosahedron(
    vertices: FloatArray,
    *,
    atol: float,
) -> FloatArray:
    pairwise_distances = squareform(pdist(vertices))
    np.fill_diagonal(pairwise_distances, np.nan)

    d_neighbor = np.zeros(len(vertices))

    for i_vertex in range(len(vertices)):
        unique, counts = np.unique(
            pairwise_distances.round(4)[i_vertex, :],
            return_counts=True,
        )
        # all vertices on the tessellated icosahedron have 6 neighbors except
        # the original 12 vertices
        n_neighbors = 5 if i_vertex < N_VERTICES_ICOSAHEDRON else 6
        idx = np.nonzero(np.cumsum(counts) == n_neighbors)[0][0]
        d_neighbor[i_vertex] = unique[idx]

    adjacency_matrix = (d_neighbor - pairwise_distances) > -atol

    neighbors = find_unique_pairs(adjacency_matrix)
    new_vertices = normalize_to_unit_length(
        vertices[neighbors[:, 0]] + vertices[neighbors[:, 1]]
    )
    return np.concatenate([vertices, new_vertices], axis=0)


def create_icosahedron_vertices() -> FloatArray:
    return normalize_to_unit_length(
        np.array(
            sorted({
                (x * x_sign, y * y_sign, z * z_sign)
                for (x, y, z), (x_sign, y_sign, z_sign) in itertools.product(
                    ((0, 1, R), (1, R, 0), (R, 0, 1)),
                    list(itertools.product([-1, +1], repeat=3)),
                )
            })
        )
    )


def create_tessellated_icosahedron_vertices(
    *,
    n_divisions: int,
    atol: float,
) -> FloatArray:
    vertices = create_icosahedron_vertices()
    for _ in range(n_divisions):
        vertices = subdivide_icosahedron(vertices, atol=atol)
    return vertices


def remove_redundant_directions(directions: FloatArray) -> FloatArray:
    return directions[
        (
            (directions[:, 2] > 0)
            | ((directions[:, 2] == 0) & (directions[:, 1] >= 0))
            | (
                (directions[:, 2] == 0)
                & (directions[:, 1] == 0)
                & (directions[:, 0] == 1)
            )
        ),
        :,
    ]


def create_discretized_directions(
    *,
    n_divisions: int = 4,
    atol: float = 1e-4,
) -> FloatArray:
    return remove_redundant_directions(
        create_tessellated_icosahedron_vertices(n_divisions=n_divisions, atol=atol)
    )
