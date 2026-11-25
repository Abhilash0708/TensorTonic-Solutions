"""
TensorTonic Exercise 02: Logistic Regression Training Loop
----------------------------------------------------------
Colab-ready script containing:
  1. Submission-ready `_sigmoid` and `train_logistic_regression` (NumPy only).
  2. Optional Colab helpers (`binary_cross_entropy`, `predict_proba`, `predict`).
  3. Verification tests matching the TensorTonic example & worked theory example.
"""

import numpy as np


# =============================================================================
# 1. TensorTonic Submission Functions (Copy ONLY this section into TensorTonic)
# =============================================================================
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise numerically stable sigmoid values without overflow warnings.
    """
    z = np.asarray(z, dtype=np.float64)
    exp_neg_abs = np.exp(-np.abs(z))
    return np.where(z >= 0.0, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs))


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 1000,
) -> tuple[np.ndarray, float]:
    """
    Trains a binary logistic regression classifier using batch gradient descent.

    Model:
        p = sigmoid(X @ w + b)
    Loss (Binary Cross-Entropy):
        L = -(1 / N) * sum(y * log(p) + (1 - y) * log(1 - p))
    Gradients:
        dw = (1 / N) * X.T @ (p - y)
        db = (1 / N) * sum(p - y)

    Args:
        X: Feature matrix of shape (N, D).
        y: Binary target labels of shape (N,) with values in {0, 1}.
        lr: Positive learning rate (default: 0.1).
        steps: Number of gradient descent iterations (default: 1000).

    Returns:
        (w, b): Learned weight vector of shape (D,) and scalar float bias.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n_samples, n_features = X_arr.shape

    if y_arr.shape[0] != n_samples:
        raise ValueError(
            f"Sample count mismatch: X has {n_samples} rows, but y has {y_arr.shape[0]} elements."
        )

    lr = float(lr)
    n_steps = int(steps)
    if lr <= 0.0:
        raise ValueError("Learning rate lr must be strictly positive.")
    if n_steps < 1:
        raise ValueError("Number of steps must be at least 1.")

    # Initialize weights as zeros and bias as 0.0
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0

    # Batch gradient descent loop
    for _ in range(n_steps):
        logits = X_arr @ w + b
        probs = _sigmoid(logits)
        error = probs - y_arr

        dw = (X_arr.T @ error) / n_samples
        db = float(np.mean(error))

        w -= lr * dw
        b -= lr * db

    return w, float(b)


# =============================================================================
# 2. Optional Colab Helpers (Colab Only - Do NOT paste into TensorTonic)
# =============================================================================
def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Computes numerically stable binary cross-entropy loss."""
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(y_prob, dtype=np.float64).reshape(-1), eps, 1.0 - eps)
    return float(-np.mean(y_t * np.log(p) + (1.0 - y_t) * np.log(1.0 - p)))


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Returns predicted positive-class probabilities."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    return _sigmoid(X_arr @ w + b)


def predict(X: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> np.ndarray:
    """Returns binary class predictions using the specified probability threshold."""
    return (predict_proba(X, w, b) >= threshold).astype(int)


# =============================================================================
# 3. Colab Verification & Edge-Case Tests
# =============================================================================
if __name__ == "__main__":
    # Test 1: TensorTonic Provided Example
    X_ex = [[0], [1], [2], [3]]
    y_ex = [0, 0, 1, 1]
    w_ex, b_ex = train_logistic_regression(X_ex, y_ex, lr=0.1, steps=500)

    assert isinstance(w_ex, np.ndarray) and w_ex.shape == (1,)
    assert isinstance(b_ex, float)
    assert np.allclose(w_ex, [2.5272], atol=1e-3)
    assert np.isclose(b_ex, -3.4307, atol=1e-3)

    probs_ex = predict_proba(X_ex, w_ex, b_ex)
    preds_ex = predict(X_ex, w_ex, b_ex)
    loss_ex = binary_cross_entropy(y_ex, probs_ex)
    assert np.array_equal(preds_ex, np.array(y_ex))

    # Test 2: Worked Theory Example (Verify Exact Step-1 Update)
    X_2d = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -1.0], [-2.0, -2.0]])
    y_2d = np.array([1, 1, 0, 0])
    w_step1, b_step1 = train_logistic_regression(X_2d, y_2d, lr=0.1, steps=1)
    assert np.allclose(w_step1, [0.075, 0.075], atol=1e-12)
    assert np.isclose(b_step1, 0.0, atol=1e-12)

    # Full training on 2D linearly separable data
    w_2d, b_2d = train_logistic_regression(X_2d, y_2d, lr=0.1, steps=1000)
    assert w_2d.shape == (2,)
    assert np.array_equal(predict(X_2d, w_2d, b_2d), y_2d)

    # Test 3: Multi-feature synthetic dataset (N=50, D=5)
    rng = np.random.default_rng(42)
    X_syn = rng.normal(size=(50, 5))
    true_w = np.array([1.5, -2.0, 0.5, 1.0, -1.2])
    y_syn = ((X_syn @ true_w + 0.3) >= 0).astype(float)
    w_syn, b_syn = train_logistic_regression(X_syn, y_syn, lr=0.2, steps=1000)
    acc_syn = np.mean(predict(X_syn, w_syn, b_syn) == y_syn)
    assert acc_syn >= 0.96

    print("All TensorTonic Logistic Regression tests passed!")
    print(f"Example Output: w = {np.round(w_ex, 4)}, b = {b_ex:.4f}")
    print(f"Example Probs:  {np.round(probs_ex, 4)} | BCE Loss: {loss_ex:.4f}")
    print(f"2D Step-1:      w = {w_step1}, b = {b_step1}")
    print(f"Synthetic Acc:  {acc_syn * 100:.1f}%")