import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


def convert_parameters_to_coordinates(
    parameters: FloatArray, *, line: tuple[FloatArray, FloatArray]
) -> FloatArray:
    a, b = line
    return a + np.multiply.outer(parameters, b)


def find_nearest_points_on_line(
    points: FloatArray,
    *,
    line: tuple[FloatArray, FloatArray],
    return_distances: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    a, b = line
    t = b @ (points - a).T / np.linalg.norm(b)
    if return_distances:
        d = np.linalg.norm(
            convert_parameters_to_coordinates(t, line=line) - points, axis=-1
        )
        return t, d
    return t
