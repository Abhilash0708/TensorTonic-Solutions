import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    Returns 0.0 if either vector has a norm of 0.
    """
    # Convert input lists to NumPy arrays for vectorized operations
    vec_a = np.array(a, dtype=float)
    vec_b = np.array(b, dtype=float)
    
    # Calculate the squared norms using dot products. 
    # This is slightly faster than using np.linalg.norm and avoids 
    # calculating the square root if we hit the zero-norm early exit.
    norm_sq_a = np.dot(vec_a, vec_a)
    norm_sq_b = np.dot(vec_b, vec_b)
    
    # Handle the zero-vector edge case to prevent division by zero
    if norm_sq_a == 0.0 or norm_sq_b == 0.0:
        return 0.0
        
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vec_a, vec_b)
    
    # Compute the final cosine similarity
    # We only take the square root once at the very end
    similarity = dot_product / np.sqrt(norm_sq_a * norm_sq_b)
    
    return float(similarity)

# --- Test Cases ---
if __name__ == "__main__":
    print(f"Example 1: {cosine_similarity([1, 2, 3], [2, 4, 6])}") # Expected: 1.0
    print(f"Example 2: {cosine_similarity([1, 0], [0, 1])}")       # Expected: 0.0
    print(f"Zero Vector: {cosine_similarity([0, 0], [5, 5])}")     # Expected: 0.0