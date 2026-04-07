import numpy as np

from ._discretize_directions import create_discretized_directions
from ._utilities import (
    FloatArray,
    compute_distances_to_line,
)


def _create_roberts_coefficients(directions: FloatArray) -> FloatArray:
    """Re-parameterize directions using Roberts' optimal line representation.

    Parameters
    ----------
    directions:
        (n_directions, 3) array of unit vectors corresponding to directions in
        3D space

    Returns
    -------
    coefficients
        (n_directions, 3, 2) array of coefficients used to convert points from
        (x', y')-space to (x, y, z)-space

    """
    b_x, b_y, b_z = np.unstack(directions, axis=-1)

    c = (b_x * b_y) / (1 + b_z)

    return np.stack(
        [
            np.stack([(1 - b_x**2 / (1 + b_z)), -c, -b_x], axis=-1),
            np.stack([-c, (1 - b_y**2 / (1 + b_z)), -b_y], axis=-1),
        ],
        axis=-1,
    )


class IterativeHoughTransform:
    def __init__(
        self,
        *,
        max_lines: int | None = None,
        min_votes: int = 2,
        n_divisions: int = 4,
        n_grid: int = 2**6,
        threshold: float | None = None,
    ) -> None:
        self.min_votes = min_votes
        self.max_lines = max_lines

        self._directions = create_discretized_directions(n_divisions=n_divisions)
        self._coefficients = _create_roberts_coefficients(self._directions)
        self._n_grid = n_grid
        self._threshold = threshold

        self._votes = np.zeros(
            (len(self._coefficients), self._n_grid, self._n_grid),
            dtype=np.int64,
        )

    def __call__(
        self,
        /,
        points: FloatArray,
    ) -> list[tuple[FloatArray, FloatArray]]:
        points = self._center_points(points)

        lines = []

        self._hough_transform(points, increase_votes=True)

        while True:
            i_direction, i_x_prime, i_y_prime = np.unravel_index(
                np.argmax(self._votes),
                self._votes.shape,
            )

            # (x', y') grid parameters:
            # x' and y' ranges: (-self._diagonal / 2, self._diagonal / 2)
            # number of intervals: (self._n_grid - 1)

            # get primed coordinates from (x', y') grid indices
            a_prime = self._diagonal * (
                np.array([i_x_prime, i_y_prime]) / (self._n_grid - 1) - 0.5
            )

            # apply self._coefficients to a' to get a in (x, y, z)-space
            a = self._coefficients[i_direction, ...] @ a_prime
            b = self._directions[i_direction]

            filter_ = self._filter_for_points_near_line(points=points, line=(a, b))
            points_on_line = points[filter_, :]

            a = np.mean(points_on_line, axis=0)
            _, _, v_t = np.linalg.svd(points_on_line - a)
            b = v_t[0, :]

            filter_ = self._filter_for_points_near_line(points=points, line=(a, b))
            points_on_line = points[filter_, :]

            points = points[~filter_]

            self._hough_transform(points_on_line, increase_votes=False)

            if len(points_on_line) >= self.min_votes:
                lines.append((a + self._offset, b))

            if ((self.max_lines is not None) and (len(lines) == self.max_lines)) or (
                len(points_on_line) < self.min_votes
            ):
                break

        return lines

    def _hough_transform(
        self,
        points: FloatArray,
        *,
        increase_votes: bool,
    ) -> None:
        xy_prime = np.einsum("dqr,pq->dpr", self._coefficients, points, optimize=True)

        # get (x', y') grid indices from primed coordinates
        indices = np.floor(self._n_grid * (xy_prime / self._diagonal + 0.5)).astype(
            np.uint32
        )

        for i_direction, indices_ in enumerate(indices):
            coordinates, count = np.unique(indices_, axis=0, return_counts=True)
            self._votes[i_direction, coordinates[:, 0], coordinates[:, 1]] += (
                1 if increase_votes else -1
            ) * count

    def _center_points(self, points: FloatArray) -> FloatArray:
        self._r_min = points.min(axis=0)
        self._r_max = points.max(axis=0)
        self._offset = 0.5 * (self._r_min + self._r_max)

        self._diagonal = np.linalg.norm(self._r_max - self._r_min)
        self._threshold = self._threshold or self._diagonal / self._n_grid

        return points - self._offset

    def _filter_for_points_near_line(
        self,
        *,
        points: FloatArray,
        line: tuple[FloatArray, FloatArray],
    ) -> FloatArray:
        d = compute_distances_to_line(points, line=line)
        return d < self._threshold
