"""
TensorTonic Exercise 03: Matrix Transpose
-----------------------------------------
Colab-ready script containing:
  1. Submission-ready `matrix_transpose` (manual vectorized slice indexing,
     no built-in transpose helpers, fast enough for 1000x1000 under 200ms).
  2. Alternative pure-NumPy 2D index-broadcasting implementation (Colab reference).
  3. Verification tests, immutability checks, and 1000x1000 benchmark.
"""

import time
import numpy as np


# =============================================================================
# 1. TensorTonic Submission Function (Copy ONLY this section into TensorTonic)
# =============================================================================
def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a new NumPy array of shape (M, N)
    without modifying the original input matrix.

    Args:
        A: 2D Python list or array-like of shape (N, M).

    Returns:
        np.ndarray: Newly allocated transposed array of shape (M, N).
    """
    arr = np.asarray(A)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, but got {arr.ndim}D input.")

    n_rows, n_cols = arr.shape
    result = np.empty((n_cols, n_rows), dtype=arr.dtype)

    # Iterate along the smaller dimension and assign full vectorized slices
    if n_rows <= n_cols:
        for i in range(n_rows):
            result[:, i] = arr[i, :]
    else:
        for j in range(n_cols):
            result[j, :] = arr[:, j]

    return result


# =============================================================================
# 2. Alternative Vectorized Fancy-Indexing Implementation (Colab Reference)
# =============================================================================
def matrix_transpose_fancy(A: list) -> np.ndarray:
    """
    Zero-loop manual coordinate-swap transpose using NumPy 2D index broadcasting:
    output[j, i] = arr[i, j].
    """
    arr = np.asarray(A)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, but got {arr.ndim}D input.")

    n_rows, n_cols = arr.shape
    row_idx = np.arange(n_rows)[None, :]  # shape (1, N)
    col_idx = np.arange(n_cols)[:, None]  # shape (M, 1)
    return arr[row_idx, col_idx]


# =============================================================================
# 3. Colab Verification, Immutability & Performance Tests
# =============================================================================
if __name__ == "__main__":
    # Example 1: Rectangular (2 x 3) -> (3 x 2)
    A1 = [[1, 2, 3], [4, 5, 6]]
    T1 = matrix_transpose(A1)
    expected_1 = np.array([[1, 4], [2, 5], [3, 6]])
    assert isinstance(T1, np.ndarray) and T1.shape == (3, 2)
    assert np.array_equal(T1, expected_1)
    assert np.array_equal(matrix_transpose_fancy(A1), expected_1)

    # Example 2: Square (2 x 2) -> (2 x 2)
    A2 = [[1, 2], [3, 4]]
    T2 = matrix_transpose(A2)
    assert T2.shape == (2, 2) and np.array_equal(T2, np.array([[1, 3], [2, 4]]))

    # Example 3: Row vector (1 x 4) -> Column vector (4 x 1)
    A3 = [[1, 2, 3, 4]]
    T3 = matrix_transpose(A3)
    assert T3.shape == (4, 1) and np.array_equal(T3, np.array([[1], [2], [3], [4]]))

    # Edge Case 1: Column vector (4 x 1) -> Row vector (1 x 4)
    A4 = [[10.5], [20.5], [30.5], [40.5]]
    T4 = matrix_transpose(A4)
    assert T4.shape == (1, 4) and np.allclose(T4, np.array([[10.5, 20.5, 30.5, 40.5]]))

    # Edge Case 2: 1 x 1 matrix & Double Transpose Involution: (A^T)^T == A
    A5 = [[42]]
    assert np.array_equal(matrix_transpose(A5), np.array([[42]]))
    assert np.array_equal(matrix_transpose(matrix_transpose(A1)), np.array(A1))

    # Edge Case 3: Immutability check (modifying output must not affect original input)
    orig_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    orig_copy = orig_np.copy()
    transposed = matrix_transpose(orig_np)
    transposed[0, 0] = -999.0
    assert np.array_equal(orig_np, orig_copy), "Original array was mutated!"

    # Edge Case 4: Verify Algebraic Product Reversal (AB)^T == B^T A^T
    rng = np.random.default_rng(7)
    X = rng.normal(size=(4, 3))
    Y = rng.normal(size=(3, 5))
    lhs = matrix_transpose(X @ Y)
    rhs = matrix_transpose(Y) @ matrix_transpose(X)
    assert np.allclose(lhs, rhs)

    # Performance Benchmark: Worst-case 1000 x 1000 constraint (limit: 200 ms)
    big_mat = rng.normal(size=(1000, 1000))
    t0 = time.perf_counter()
    big_T = matrix_transpose(big_mat)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert big_T.shape == (1000, 1000)
    assert np.array_equal(big_T, big_mat.T)

    print("All TensorTonic Matrix Transpose tests passed!")
    print(f"Example 1 Output:\n{T1}")
    print(f"1000 x 1000 Transpose Runtime: {elapsed_ms:.2f} ms (limit: 200 ms)")