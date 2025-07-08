# Algebra for Machine Learning - Cheat Sheet

## 1. Linear Algebra Fundamentals

### Vectors
A vector is an ordered list of numbers representing magnitude and direction.

**Python Implementation:**
```python
import numpy as np

# Create vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector operations
addition = v1 + v2          # [5, 7, 9]
subtraction = v1 - v2       # [-3, -3, -3]
scalar_mult = 2 * v1        # [2, 4, 6]
magnitude = np.linalg.norm(v1)  # √(1² + 2² + 3²) = √14
```

### Dot Product
The dot product measures similarity between vectors.

**Formula:** `a · b = |a| |b| cos(θ)`

```python
# Dot product
dot_product = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32

# Alternative syntax
dot_product = v1 @ v2
```

### Matrix Operations

```python
# Create matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix operations
addition = A + B
multiplication = A @ B  # Matrix multiplication
transpose = A.T
inverse = np.linalg.inv(A)
determinant = np.linalg.det(A)
```

## 2. Eigenvalues and Eigenvectors

Essential for PCA, dimensionality reduction, and understanding data variance.

**Definition:** For matrix A, vector v is an eigenvector with eigenvalue λ if: `Av = λv`

```python
# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Verify: A @ eigenvector = eigenvalue * eigenvector
v = eigenvectors[:, 0]  # First eigenvector
λ = eigenvalues[0]      # First eigenvalue
print("Av =", A @ v)
print("λv =", λ * v)
```

## 3. Matrix Decompositions

### Singular Value Decomposition (SVD)
Used in PCA, recommender systems, and data compression.

**Formula:** `A = UΣV^T`

```python
# SVD decomposition
U, sigma, Vt = np.linalg.svd(A)

# Reconstruct original matrix
A_reconstructed = U @ np.diag(sigma) @ Vt
print("Original A:\n", A)
print("Reconstructed A:\n", A_reconstructed)
```

### Cholesky Decomposition
For positive definite matrices (covariance matrices).

```python
# Create positive definite matrix
cov_matrix = np.array([[4, 2], [2, 3]])

# Cholesky decomposition
L = np.linalg.cholesky(cov_matrix)
print("L:\n", L)
print("L @ L.T =\n", L @ L.T)  # Should equal cov_matrix
```

## 4. Norms and Distance Metrics

### Vector Norms
```python
v = np.array([3, 4, 5])

# L1 norm (Manhattan distance)
l1_norm = np.linalg.norm(v, 1)  # |3| + |4| + |5| = 12

# L2 norm (Euclidean distance)
l2_norm = np.linalg.norm(v, 2)  # √(3² + 4² + 5²) = √50

# L∞ norm (Maximum norm)
linf_norm = np.linalg.norm(v, np.inf)  # max(|3|, |4|, |5|) = 5
```

### Distance Between Vectors
```python
from scipy.spatial.distance import euclidean, manhattan, cosine

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

euclidean_dist = euclidean(v1, v2)
manhattan_dist = manhattan(v1, v2)
cosine_dist = cosine(v1, v2)  # 1 - cosine_similarity
```

## 5. Quadratic Forms and Optimization

### Quadratic Forms
Common in loss functions and optimization.

**Formula:** `x^T A x`

```python
# Quadratic form
x = np.array([1, 2])
A = np.array([[2, 1], [1, 3]])

quadratic_form = x.T @ A @ x
print("Quadratic form:", quadratic_form)  # 1*2*1 + 1*1*2 + 2*1*1 + 2*3*2 = 16
```

### Gradient and Hessian
```python
# For function f(x) = x^T A x + b^T x + c
# Gradient: ∇f = 2Ax + b
# Hessian: H = 2A

def quadratic_function(x, A, b, c):
    return x.T @ A @ x + b.T @ x + c

def gradient(x, A, b):
    return 2 * A @ x + b

def hessian(A):
    return 2 * A

# Example
x = np.array([1, 2])
A = np.array([[1, 0], [0, 1]])
b = np.array([1, 1])
c = 0

grad = gradient(x, A, b)
hess = hessian(A)
```

## 6. Systems of Linear Equations

### Solving Ax = b
```python
# System: 2x + 3y = 7, x + 4y = 6
A = np.array([[2, 3], [1, 4]])
b = np.array([7, 6])

# Method 1: Direct solve
x = np.linalg.solve(A, b)

# Method 2: Using inverse
x = np.linalg.inv(A) @ b

# Method 3: Least squares (for overdetermined systems)
x = np.linalg.lstsq(A, b, rcond=None)[0]
```

## 7. Matrix Factorizations for ML

### QR Decomposition
```python
# QR decomposition
Q, R = np.linalg.qr(A)

# Q is orthogonal, R is upper triangular
print("Q @ Q.T =\n", Q @ Q.T)  # Should be identity
print("Q @ R =\n", Q @ R)      # Should equal A
```

### LU Decomposition
```python
from scipy.linalg import lu

# LU decomposition
P, L, U = lu(A)

# P is permutation matrix, L is lower triangular, U is upper triangular
print("P @ L @ U =\n", P @ L @ U)  # Should equal A
```

## 8. Practical ML Applications

### Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample data
X = np.random.randn(100, 5)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Principal components:\n", pca.components_)
```

### Linear Regression (Normal Equation)
```python
# Linear regression: y = Xβ + ε
# Solution: β = (X^T X)^(-1) X^T y

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)
true_beta = np.array([1.5, -2.0, 0.5])
y = X @ true_beta + 0.1 * np.random.randn(100)

# Add bias term
X_with_bias = np.column_stack([np.ones(100), X])

# Normal equation
XtX_inv = np.linalg.inv(X_with_bias.T @ X_with_bias)
beta_hat = XtX_inv @ X_with_bias.T @ y

print("Estimated coefficients:", beta_hat)
```

### Covariance Matrix
```python
# Covariance matrix calculation
def covariance_matrix(X):
    # X should be centered (mean subtracted)
    n = X.shape[0]
    return (X.T @ X) / (n - 1)

# Center the data
X_centered = X - np.mean(X, axis=0)
cov_matrix = covariance_matrix(X_centered)

# Or use numpy
cov_matrix_np = np.cov(X.T)
```

## 9. Common Matrix Properties

### Positive Definite Matrices
```python
def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Test matrix
A = np.array([[2, 1], [1, 2]])
print("Is positive definite:", is_positive_definite(A))

# Eigenvalue test (all eigenvalues > 0)
eigenvals = np.linalg.eigvals(A)
print("Eigenvalues:", eigenvals)
print("All positive:", all(eigenvals > 0))
```

### Orthogonal Matrices
```python
def is_orthogonal(Q, tolerance=1e-10):
    return np.allclose(Q @ Q.T, np.eye(Q.shape[0]), atol=tolerance)

# Create orthogonal matrix using QR decomposition
A = np.random.randn(3, 3)
Q, R = np.linalg.qr(A)
print("Q is orthogonal:", is_orthogonal(Q))
```

## 10. Numerical Stability Tips

### Condition Number
```python
# Check condition number (for numerical stability)
cond_number = np.linalg.cond(A)
print("Condition number:", cond_number)

# Rule of thumb: cond(A) > 1e12 indicates numerical instability
if cond_number > 1e12:
    print("Matrix is ill-conditioned!")
```

### Regularization
```python
# Ridge regression (L2 regularization)
def ridge_regression(X, y, alpha=0.01):
    # β = (X^T X + αI)^(-1) X^T y
    n_features = X.shape[1]
    I = np.eye(n_features)
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return beta

# Usage
X = np.random.randn(50, 10)
y = np.random.randn(50)
beta_ridge = ridge_regression(X, y, alpha=0.1)
```

## Key Formulas Summary

| Operation | Formula | NumPy Code |
|-----------|---------|------------|
| Dot Product | a · b = Σ(aᵢbᵢ) | `np.dot(a, b)` |
| Matrix Multiplication | (AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ | `A @ B` |
| Eigendecomposition | A = QΛQ⁻¹ | `np.linalg.eig(A)` |
| SVD | A = UΣVᵀ | `np.linalg.svd(A)` |
| Normal Equation | β = (XᵀX)⁻¹Xᵀy | `np.linalg.solve(X.T @ X, X.T @ y)` |
| L2 Norm | ‖x‖₂ = √(Σxᵢ²) | `np.linalg.norm(x)` |

## Performance Tips

1. **Use `@` for matrix multiplication** instead of `np.dot()` for better readability
2. **Prefer `np.linalg.solve(A, b)`** over `np.linalg.inv(A) @ b` for better numerical stability
3. **Use `scipy.linalg`** for advanced linear algebra operations
4. **Consider using `numpy.einsum`** for complex tensor operations
5. **Always check condition numbers** before matrix inversion