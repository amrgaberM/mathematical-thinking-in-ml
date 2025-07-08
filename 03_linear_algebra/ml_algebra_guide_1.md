# Complete Algebra for Machine Learning - Study Guide

## Table of Contents
1. [Linear Algebra Fundamentals](#linear-algebra-fundamentals)
2. [Vectors and Vector Operations](#vectors-and-vector-operations)
3. [Matrices and Matrix Operations](#matrices-and-matrix-operations)
4. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
5. [Matrix Decompositions](#matrix-decompositions)
6. [Calculus for ML](#calculus-for-ml)
7. [Probability and Statistics](#probability-and-statistics)
8. [Optimization Theory](#optimization-theory)
9. [Practical ML Applications](#practical-ml-applications)

---

## 1. Linear Algebra Fundamentals

### What is Linear Algebra?
Linear algebra is the branch of mathematics dealing with vectors, vector spaces, linear transformations, and systems of linear equations. In machine learning, it's fundamental because:

- **Data representation**: Data is represented as vectors and matrices
- **Feature transformations**: Linear transformations modify feature spaces
- **Model parameters**: Weights and biases are vectors/matrices
- **Optimization**: Gradient descent uses vector calculus

### Vector Spaces
A vector space is a collection of objects (vectors) that can be added together and multiplied by scalars while satisfying certain axioms.

**Properties of Vector Spaces:**
- Closure under addition and scalar multiplication
- Existence of zero vector
- Existence of additive inverses
- Associativity and commutativity of addition
- Distributivity of scalar multiplication

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: R² vector space
# Vectors in 2D space
v1 = np.array([3, 4])
v2 = np.array([1, 2])
scalar = 2

# Vector addition
v_sum = v1 + v2
print(f"v1 + v2 = {v_sum}")

# Scalar multiplication
v_scaled = scalar * v1
print(f"2 * v1 = {v_scaled}")

# Zero vector
zero_vector = np.zeros(2)
print(f"Zero vector: {zero_vector}")

# Visualizing vectors
plt.figure(figsize=(10, 6))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='red', label='v2')
plt.quiver(0, 0, v_sum[0], v_sum[1], angles='xy', scale_units='xy', scale=1, color='green', label='v1+v2')
plt.xlim(-1, 6)
plt.ylim(-1, 7)
plt.grid(True)
plt.legend()
plt.title('Vector Addition in R²')
plt.show()
```

---

## 2. Vectors and Vector Operations

### Vector Basics
A vector is an ordered list of numbers. In ML, vectors represent:
- **Feature vectors**: Individual data points
- **Weight vectors**: Model parameters
- **Gradient vectors**: Direction of steepest increase

### Vector Operations

#### Dot Product (Inner Product)
The dot product measures the similarity between two vectors and projects one onto another.

**Formula**: `a · b = Σ(aᵢ × bᵢ) = |a| |b| cos(θ)`

```python
# Dot product implementation and applications
def dot_product(a, b):
    """Calculate dot product of two vectors"""
    return np.sum(a * b)

# Example vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Built-in dot product
dot_builtin = np.dot(a, b)
print(f"Dot product: {dot_builtin}")

# Manual calculation
dot_manual = dot_product(a, b)
print(f"Manual dot product: {dot_manual}")

# Geometric interpretation
def vector_angle(a, b):
    """Calculate angle between two vectors"""
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(cos_angle) * 180 / np.pi

angle = vector_angle(a, b)
print(f"Angle between vectors: {angle:.2f} degrees")

# Application: Cosine similarity (used in recommendation systems)
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example: Document similarity
doc1 = np.array([3, 1, 4, 1, 5])  # Word frequencies
doc2 = np.array([2, 1, 3, 1, 4])  # Word frequencies

similarity = cosine_similarity(doc1, doc2)
print(f"Document similarity: {similarity:.3f}")
```

#### Cross Product
The cross product produces a vector perpendicular to both input vectors (only defined in 3D).

```python
# Cross product (3D vectors only)
def cross_product_3d(a, b):
    """Calculate cross product of two 3D vectors"""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

a_3d = np.array([1, 2, 3])
b_3d = np.array([4, 5, 6])

cross_manual = cross_product_3d(a_3d, b_3d)
cross_builtin = np.cross(a_3d, b_3d)

print(f"Cross product (manual): {cross_manual}")
print(f"Cross product (built-in): {cross_builtin}")

# Verify perpendicularity
print(f"a · (a × b) = {np.dot(a_3d, cross_builtin)}")  # Should be 0
print(f"b · (a × b) = {np.dot(b_3d, cross_builtin)}")  # Should be 0
```

#### Vector Norms
Norms measure the "size" or "length" of a vector.

```python
# Different types of norms
def calculate_norms(vector):
    """Calculate various norms of a vector"""
    # L1 norm (Manhattan distance)
    l1_norm = np.sum(np.abs(vector))
    
    # L2 norm (Euclidean distance)
    l2_norm = np.sqrt(np.sum(vector**2))
    
    # L∞ norm (Maximum norm)
    l_inf_norm = np.max(np.abs(vector))
    
    # P-norm (general case)
    def p_norm(v, p):
        return np.sum(np.abs(v)**p)**(1/p)
    
    return {
        'L1': l1_norm,
        'L2': l2_norm,
        'L∞': l_inf_norm,
        'L3': p_norm(vector, 3)
    }

test_vector = np.array([3, -4, 5])
norms = calculate_norms(test_vector)

for norm_name, norm_value in norms.items():
    print(f"{norm_name} norm: {norm_value:.3f}")

# Built-in numpy norms
print(f"NumPy L1 norm: {np.linalg.norm(test_vector, 1):.3f}")
print(f"NumPy L2 norm: {np.linalg.norm(test_vector, 2):.3f}")
print(f"NumPy L∞ norm: {np.linalg.norm(test_vector, np.inf):.3f}")
```

### Linear Independence and Basis
Vectors are linearly independent if no vector can be written as a linear combination of the others.

```python
# Check linear independence
def is_linearly_independent(vectors):
    """Check if a set of vectors is linearly independent"""
    # Stack vectors as columns
    matrix = np.column_stack(vectors)
    
    # Calculate rank
    rank = np.linalg.matrix_rank(matrix)
    
    # Linearly independent if rank equals number of vectors
    return rank == len(vectors)

# Example vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v3 = np.array([7, 8, 9])  # This is linearly dependent on v1 and v2

vectors = [v1, v2, v3]
print(f"Are vectors linearly independent? {is_linearly_independent(vectors)}")

# Check with independent vectors
v3_independent = np.array([1, 0, 1])
vectors_independent = [v1, v2, v3_independent]
print(f"Are modified vectors linearly independent? {is_linearly_independent(vectors_independent)}")
```

---

## 3. Matrices and Matrix Operations

### Matrix Fundamentals
A matrix is a rectangular array of numbers arranged in rows and columns. In ML:
- **Data matrices**: Rows are samples, columns are features
- **Weight matrices**: Parameters in neural networks
- **Transformation matrices**: Feature transformations

### Basic Matrix Operations

```python
# Matrix creation and basic operations
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Matrix addition
C = A + B
print("\nA + B:")
print(C)

# Matrix subtraction
D = A - B
print("\nA - B:")
print(D)

# Element-wise multiplication (Hadamard product)
E = A * B
print("\nA * B (element-wise):")
print(E)

# Matrix multiplication
F = A @ B  # or np.dot(A, B)
print("\nA @ B (matrix multiplication):")
print(F)
```

### Matrix Multiplication Deep Dive
Matrix multiplication is fundamental to ML operations.

```python
# Understanding matrix multiplication
def matrix_multiply_manual(A, B):
    """Manual implementation of matrix multiplication"""
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    
    if cols_A != rows_B:
        raise ValueError("Matrix dimensions don't match for multiplication")
    
    result = np.zeros((rows_A, cols_B))
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i, j] += A[i, k] * B[k, j]
    
    return result

# Example: Neural network forward pass
def neural_network_layer(X, W, b):
    """
    Forward pass through a neural network layer
    X: input matrix (samples x features)
    W: weight matrix (features x neurons)
    b: bias vector (neurons,)
    """
    return X @ W + b

# Example data
X = np.array([[1, 2, 3],     # Sample 1
              [4, 5, 6],     # Sample 2
              [7, 8, 9]])    # Sample 3

W = np.array([[0.1, 0.2],    # Weights for feature 1
              [0.3, 0.4],    # Weights for feature 2
              [0.5, 0.6]])   # Weights for feature 3

b = np.array([0.1, 0.2])     # Biases

output = neural_network_layer(X, W, b)
print("Neural network layer output:")
print(output)
```

### Matrix Properties

```python
# Important matrix properties
def matrix_properties(A):
    """Calculate various matrix properties"""
    properties = {}
    
    # Shape
    properties['shape'] = A.shape
    
    # Transpose
    properties['transpose'] = A.T
    
    # Trace (sum of diagonal elements)
    if A.shape[0] == A.shape[1]:  # Square matrix
        properties['trace'] = np.trace(A)
    
    # Determinant (for square matrices)
    if A.shape[0] == A.shape[1]:
        properties['determinant'] = np.linalg.det(A)
    
    # Rank
    properties['rank'] = np.linalg.matrix_rank(A)
    
    # Norm
    properties['frobenius_norm'] = np.linalg.norm(A, 'fro')
    
    return properties

# Example matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])  # Made it non-singular

props = matrix_properties(A)
for prop_name, prop_value in props.items():
    print(f"{prop_name}: {prop_value}")
```

### Matrix Inverse
The inverse of a matrix A is denoted A⁻¹ and satisfies AA⁻¹ = I.

```python
# Matrix inverse and applications
def matrix_inverse_applications():
    """Demonstrate matrix inverse applications"""
    
    # Create an invertible matrix
    A = np.array([[2, 1, 1],
                  [1, 3, 2],
                  [1, 0, 0]])
    
    print("Original matrix A:")
    print(A)
    
    # Calculate inverse
    A_inv = np.linalg.inv(A)
    print("\nInverse of A:")
    print(A_inv)
    
    # Verify A * A^(-1) = I
    identity_check = A @ A_inv
    print("\nA @ A^(-1) (should be identity):")
    print(identity_check)
    
    # Application: Solving linear system Ax = b
    b = np.array([1, 2, 3])
    x_inv = A_inv @ b
    print(f"\nSolving Ax = b using inverse: x = {x_inv}")
    
    # Verify solution
    print(f"Verification A @ x = {A @ x_inv}")
    
    # Better approach: Use solve (more numerically stable)
    x_solve = np.linalg.solve(A, b)
    print(f"Using np.linalg.solve: x = {x_solve}")
    
    return A, A_inv

matrix_inverse_applications()
```

### Special Matrices

```python
# Special types of matrices important in ML
def create_special_matrices():
    """Create and demonstrate special matrices"""
    
    # Identity matrix
    I = np.eye(3)
    print("Identity matrix:")
    print(I)
    
    # Zero matrix
    Z = np.zeros((3, 3))
    print("\nZero matrix:")
    print(Z)
    
    # Diagonal matrix
    D = np.diag([1, 2, 3])
    print("\nDiagonal matrix:")
    print(D)
    
    # Symmetric matrix
    A = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])
    print("\nSymmetric matrix:")
    print(A)
    print(f"Is symmetric: {np.allclose(A, A.T)}")
    
    # Orthogonal matrix (columns are orthonormal)
    # Using QR decomposition to create one
    random_matrix = np.random.randn(3, 3)
    Q, R = np.linalg.qr(random_matrix)
    print("\nOrthogonal matrix:")
    print(Q)
    print(f"Q @ Q.T (should be identity):")
    print(Q @ Q.T)
    
    return I, Z, D, A, Q

create_special_matrices()
```

---

## 4. Eigenvalues and Eigenvectors

### Theory
For a square matrix A, an eigenvector v is a non-zero vector such that Av = λv, where λ is the corresponding eigenvalue.

**Importance in ML:**
- Principal Component Analysis (PCA)
- Spectral clustering
- Stability analysis of optimization algorithms
- Understanding covariance matrices

```python
# Eigenvalue and eigenvector computation
def eigenvalue_analysis():
    """Comprehensive eigenvalue and eigenvector analysis"""
    
    # Create a symmetric matrix (has real eigenvalues)
    A = np.array([[4, 2, 1],
                  [2, 3, 0],
                  [1, 0, 2]])
    
    print("Matrix A:")
    print(A)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("\nEigenvalues:")
    print(eigenvalues)
    
    print("\nEigenvectors:")
    print(eigenvectors)
    
    # Verify the eigenvalue equation: Av = λv
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        Av = A @ vec
        lv = val * vec
        print(f"\nEigenvalue {i+1}: λ = {val:.3f}")
        print(f"Av = {Av}")
        print(f"λv = {lv}")
        print(f"Av ≈ λv: {np.allclose(Av, lv)}")
    
    return A, eigenvalues, eigenvectors

eigenvalue_analysis()
```

### Eigendecomposition
A matrix can be decomposed as A = QΛQ⁻¹, where Q contains eigenvectors and Λ contains eigenvalues.

```python
# Eigendecomposition and reconstruction
def eigendecomposition_demo():
    """Demonstrate eigendecomposition and matrix reconstruction"""
    
    # Symmetric matrix for clear demonstration
    A = np.array([[3, 1, 1],
                  [1, 3, 1],
                  [1, 1, 3]])
    
    print("Original matrix A:")
    print(A)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Create diagonal matrix of eigenvalues
    Lambda = np.diag(eigenvalues)
    Q = eigenvectors
    
    print("\nEigenvalue matrix Λ:")
    print(Lambda)
    
    print("\nEigenvector matrix Q:")
    print(Q)
    
    # Reconstruct original matrix: A = Q Λ Q^(-1)
    A_reconstructed = Q @ Lambda @ np.linalg.inv(Q)
    
    print("\nReconstructed matrix A:")
    print(A_reconstructed)
    
    print(f"\nReconstruction error: {np.linalg.norm(A - A_reconstructed)}")
    
    # For symmetric matrices, Q is orthogonal, so Q^(-1) = Q^T
    if np.allclose(A, A.T):
        A_reconstructed_symmetric = Q @ Lambda @ Q.T
        print(f"Symmetric reconstruction error: {np.linalg.norm(A - A_reconstructed_symmetric)}")
    
    return A, eigenvalues, eigenvectors

eigendecomposition_demo()
```

### Application: Principal Component Analysis (PCA)

```python
# PCA implementation using eigendecomposition
def pca_from_scratch(X, n_components=2):
    """
    Implement PCA using eigendecomposition
    X: data matrix (samples x features)
    n_components: number of principal components
    """
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    principal_components = eigenvectors[:, :n_components]
    
    # Transform data
    X_pca = X_centered @ principal_components
    
    # Explained variance ratio
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    
    return X_pca, principal_components, explained_variance_ratio

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 4)  # 100 samples, 4 features
X[:, 1] = X[:, 0] + 0.5 * np.random.randn(100)  # Create some correlation

# Apply PCA
X_pca, components, var_ratio = pca_from_scratch(X, n_components=2)

print("Original data shape:", X.shape)
print("PCA data shape:", X_pca.shape)
print("Explained variance ratio:", var_ratio[:2])
print("Cumulative explained variance:", np.cumsum(var_ratio[:2]))

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Original Data (First 2 Features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.title('PCA Transformed Data')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.tight_layout()
plt.show()
```

---

## 5. Matrix Decompositions

### Singular Value Decomposition (SVD)
SVD decomposes any matrix A into A = UΣV^T, where U and V are orthogonal matrices and Σ is diagonal.

```python
# SVD implementation and applications
def svd_analysis():
    """Demonstrate SVD and its applications"""
    
    # Create a sample matrix
    A = np.array([[3, 1, 1],
                  [1, 3, 1],
                  [1, 1, 3],
                  [0, 0, 1]])
    
    print("Original matrix A:")
    print(A)
    print(f"Shape: {A.shape}")
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(A)
    
    print(f"\nU shape: {U.shape}")
    print(f"s shape: {s.shape}")
    print(f"Vt shape: {Vt.shape}")
    
    # Reconstruct matrix
    # Need to pad s with zeros for non-square matrices
    Sigma = np.zeros((A.shape[0], A.shape[1]))
    Sigma[:min(A.shape), :min(A.shape)] = np.diag(s)
    
    A_reconstructed = U @ Sigma @ Vt
    
    print("\nReconstructed matrix:")
    print(A_reconstructed)
    print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")
    
    # Low-rank approximation
    def low_rank_approximation(U, s, Vt, k):
        """Create rank-k approximation"""
        return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    
    # Rank-2 approximation
    A_rank2 = low_rank_approximation(U, s, Vt, 2)
    print(f"\nRank-2 approximation error: {np.linalg.norm(A - A_rank2)}")
    
    return U, s, Vt

svd_analysis()
```

### QR Decomposition
QR decomposition factorizes a matrix A into A = QR, where Q is orthogonal and R is upper triangular.

```python
# QR decomposition and applications
def qr_decomposition_demo():
    """Demonstrate QR decomposition and its applications"""
    
    # Create a matrix
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 10],
                  [11, 12, 13]], dtype=float)
    
    print("Original matrix A:")
    print(A)
    
    # QR decomposition
    Q, R = np.linalg.qr(A)
    
    print("\nQ matrix (orthogonal):")
    print(Q)
    
    print("\nR matrix (upper triangular):")
    print(R)
    
    # Verify orthogonality of Q
    print(f"\nQ.T @ Q (should be identity):")
    print(Q.T @ Q)
    
    # Verify reconstruction
    A_reconstructed = Q @ R
    print(f"\nReconstruction error: {np.linalg.norm(A - A_reconstructed)}")
    
    # Application: Solving least squares problems
    # For overdetermined system Ax = b, solution is x = R^(-1) @ Q.T @ b
    b = np.array([1, 2, 3, 4])
    
    # Solve using QR
    x_qr = np.linalg.solve(R, Q.T @ b)
    print(f"\nLeast squares solution using QR: {x_qr}")
    
    # Compare with numpy's least squares
    x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"NumPy least squares solution: {x_lstsq}")
    
    return Q, R

qr_decomposition_demo()
```

### Cholesky Decomposition
For positive definite matrices, Cholesky decomposition gives A = LL^T, where L is lower triangular.

```python
# Cholesky decomposition
def cholesky_decomposition_demo():
    """Demonstrate Cholesky decomposition"""
    
    # Create a positive definite matrix
    A = np.array([[4, 2, 1],
                  [2, 3, 0.5],
                  [1, 0.5, 2]])
    
    print("Original matrix A (positive definite):")
    print(A)
    
    # Check if positive definite
    eigenvalues = np.linalg.eigvals(A)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Is positive definite: {np.all(eigenvalues > 0)}")
    
    # Cholesky decomposition
    L = np.linalg.cholesky(A)
    
    print("\nCholesky factor L:")
    print(L)
    
    # Verify A = L @ L.T
    A_reconstructed = L @ L.T
    print(f"\nReconstruction error: {np.linalg.norm(A - A_reconstructed)}")
    
    # Application: Generating correlated random variables
    # If X ~ N(0, I), then L @ X ~ N(0, A)
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(3, n_samples)
    
    # Transform to have covariance A
    Y = L @ X
    
    # Check empirical covariance
    empirical_cov = np.cov(Y)
    print(f"\nEmpirical covariance (should be close to A):")
    print(empirical_cov)
    
    return L

cholesky_decomposition_demo()
```

---

## 6. Calculus for ML

### Derivatives and Gradients
Derivatives measure how functions change with respect to their inputs. In ML, we use gradients to optimize loss functions.

```python
# Numerical differentiation and gradient computation
def numerical_derivative(f, x, h=1e-7):
    """Compute numerical derivative using finite differences"""
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient(f, x, h=1e-7):
    """Compute numerical gradient for multivariate functions"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Example functions
def f_univariate(x):
    """Simple univariate function: f(x) = x^2 + 2x + 1"""
    return x**2 + 2*x + 1

def f_multivariate(x):
    """Multivariate function: f(x) = x1^2 + x2^2 + x1*x2"""
    return x[0]**2 + x[1]**2 + x[0]*x[1]

# Compute derivatives
x_point = 2.0
derivative = numerical_derivative(f_univariate, x_point)
print(f"Derivative of f(x) at x={x_point}: {derivative}")
print(f"Analytical derivative: {2*x_point + 2}")

# Compute gradients
x_point = np.array([1.0, 2.0])
gradient = numerical_gradient(f_multivariate, x_point)
print(f"\nGradient of f(x) at x={x_point}: {gradient}")
print(f"Analytical gradient: [{2*x_point[0] + x_point[1]}, {2*x_point[1] + x_point[0]}]")
```

### Chain Rule
The chain rule is fundamental for backpropagation in neural networks.

```python
# Chain rule implementation
class ComputationGraph:
    """Simple computation graph for automatic differentiation"""
    
    def __init__(self):
        self.nodes = []
        self.values = {}
        self.gradients = {}
    
    def add_node(self, name, value, grad_fn=None):
        """Add a node to the computation graph"""
        self.nodes.append(name)
        self.values[name] = value
        self.gradients[name] = 0.0
        if grad_fn:
            self.grad_fn = grad_fn
    
    def forward(self, x):
        """Forward pass through a simple neural network"""
        # f(x) = (x^2 + 1)^3
        
        # Step 1: u = x^2
        u = x**2
        self.values['u'] = u
        
        # Step 2: v = u + 1
        v = u + 1
        self.values['v'] = v
        
        # Step 3: y = v^3
        y = v**3
        self.values['y'] = y
        
        return y
    
    def backward(self, x):
        """Backward pass using chain rule"""
        # dy/dy = 