"""
TensorTonic Exercise: Implement Sigmoid in NumPy
------------------------------------------------
Implements a fully vectorized, numerically stable sigmoid activation function
using only NumPy. Handles scalars, 1D/ND Python lists, and NumPy arrays while
preventing floating-point overflow for extreme negative/positive logits.
"""

import numpy as np


def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list/array.

    Uses the numerically stable identity with z = exp(-|x|):
        - For x >= 0: 1 / (1 + exp(-x))  = 1 / (1 + z)
        - For x <  0: exp(x) / (1 + exp(x)) = z / (1 + z)

    Because -|x| <= 0 always holds, exp(-|x|) is bounded in [0, 1], preventing
    overflow warnings while preserving full float64 precision for large |x|.

    Args:
        x: A numeric scalar (int, float, np.number), a Python list / nested list,
           or a NumPy array of numeric values.

    Returns:
        float: Native Python float if x is a scalar.
        np.ndarray: NumPy array of float64 values matching the shape of x otherwise.

    Raises:
        TypeError: If x cannot be converted into a numeric float64 NumPy array.
        ValueError: If x has ragged/inhomogeneous nested list dimensions.
    """
    try:
        # Convert input (scalar, list, nested list, or ndarray) to float64 array
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Input x must be a numeric scalar or a rectangular list/array of numbers."
        ) from exc

    # Compute exp(-|x|) once: always in [0, 1], so it never overflows
    z = np.exp(-np.abs(arr))

    # Select stable branch element-wise without Python loops:
    #   x >= 0 -> 1 / (1 + z)
    #   x <  0 -> z / (1 + z)
    result = np.where(arr >= 0.0, 1.0 / (1.0 + z), z / (1.0 + z))

    # Return a plain Python float when the input was a 0-D scalar
    if arr.ndim == 0:
        return float(result)

    return result


# ---------------------------------------------------------------------------
# Local Verification Suite (matches TensorTonic examples + edge cases)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Example 1: 1D list
    out_1d = sigmoid([0, 2, -2])
    expected_1d = np.array([0.5, 0.88079708, 0.11920292])
    assert isinstance(out_1d, np.ndarray) and out_1d.dtype == np.float64
    assert np.allclose(out_1d, expected_1d, atol=1e-8)

    # 2. Example 2: Scalar input -> must return native Python float
    out_scalar = sigmoid(0)
    assert isinstance(out_scalar, float) and np.isclose(out_scalar, 0.5)

    # 3. Example 3: 2D nested list -> must preserve shape (2, 2)
    out_2d = sigmoid([[-1, 0], [1, 2]])
    expected_2d = np.array([[0.26894142, 0.5], [0.73105858, 0.88079708]])
    assert isinstance(out_2d, np.ndarray) and out_2d.shape == (2, 2)
    assert np.allclose(out_2d, expected_2d, atol=1e-8)

    # 4. Numerical stability edge cases: extreme logits & infinities (no overflow)
    out_extreme = sigmoid([-1000.0, -50.0, 50.0, 1000.0, -np.inf, np.inf])
    assert out_extreme[0] == 0.0
    assert np.isclose(out_extreme[1], np.exp(-50.0), rtol=1e-12)
    assert np.isclose(out_extreme[2], 1.0)
    assert out_extreme[3] == 1.0
    assert out_extreme[4] == 0.0
    assert out_extreme[5] == 1.0

    # 5. Empty list edge case
    out_empty = sigmoid([])
    assert isinstance(out_empty, np.ndarray) and out_empty.shape == (0,)

    print("All TensorTonic sigmoid tests passed!")
    print("1D Output:", out_1d)
    print("Scalar Output:", out_scalar, type(out_scalar))
    print("2D Output:\n", out_2d)