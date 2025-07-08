# Complete Algebra for Machine Learning - Theory & Practice
# ===========================================================

"""
This notebook provides a comprehensive learning material combining mathematical theory
with practical implementations for machine learning applications.

Learning Objectives:
- Understand the mathematical foundations behind ML algorithms
- Implement algebraic concepts from scratch
- See how these concepts apply to real ML problems
- Build intuition through visualizations and examples

Prerequisites: Basic Python knowledge
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
plt.style.use('seaborn-v0_8')

print("🎓 Complete Algebra for Machine Learning - Theory & Practice")
print("=" * 65)

# =============================================================================
# CHAPTER 1: VECTORS - THE BUILDING BLOCKS OF ML
# =============================================================================

print("\n📚 CHAPTER 1: VECTORS - THE BUILDING BLOCKS OF ML")
print("=" * 55)

# 1.1 Vector Theory and Intuition
print("\n1.1 Vector Theory and Intuition")
print("-" * 32)

"""
THEORY:
A vector is a mathematical object with both magnitude and direction.
In machine learning context:
- Data points are vectors in feature space
- Each feature is a dimension
- Algorithms operate on these high-dimensional vectors

Mathematical Definition:
A vector v ∈ ℝⁿ is an ordered n-tuple: v = (v₁, v₂, ..., vₙ)

ML Applications:
- Feature vectors: [age, income, education_years, ...]
- Word embeddings: [0.1, -0.3, 0.7, ..., 0.2]
- Neural network weights and activations
"""

# Practical Example: Customer Data as Vectors
customers = np.array([
    [25, 50000, 16, 1],    # [age, income, education, has_car]
    [35, 75000, 18, 1],
    [28, 45000, 14, 0],
    [42, 90000, 20, 1],
    [22, 30000, 12, 0]
])

print("Customer Data as Vectors:")
print("Features: [age, income, education_years, has_car]")
for i, customer in enumerate(customers):
    print(f"Customer {i+1}: {customer}")

# Visualizing high-dimensional data in 2D
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 2D visualization of age vs income
ax1.scatter(customers[:, 0], customers[:, 1], s=100, alpha=0.7)
ax1.set_xlabel('Age')
ax1.set_ylabel('Income')
ax1.set_title('Customers in 2D Space (Age vs Income)')
ax1.grid(True, alpha=0.3)

# Add customer labels
for i, (age, income) in enumerate(customers[:, :2]):
    ax1.annotate(f'C{i+1}', (age, income), xytext=(5, 5), textcoords='offset points')

# Vector representation
origin = np.array([0, 0])
for i, customer in enumerate(customers[:, :2]):
    ax2.arrow(origin[0], origin[1], customer[0], customer[1], 
             head_width=1.5, head_length=2000, fc=f'C{i}', ec=f'C{i}', alpha=0.7)
    ax2.text(customer[0]/2, customer[1]/2, f'C{i+1}', fontsize=10)

ax2.set_xlabel('Age')
ax2.set_ylabel('Income')
ax2.set_title('Customers as Vectors from Origin')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 45)
ax2.set_ylim(0, 95000)

plt.tight_layout()
plt.show()

# 1.2 Vector Operations in ML Context
print("\n1.2 Vector Operations in ML Context")
print("-" * 35)

"""
THEORY:
Vector operations are fundamental to ML algorithms:

1. Addition: v + w = (v₁ + w₁, v₂ + w₂, ..., vₙ + wₙ)
   - Used in: Gradient updates, ensemble methods
   
2. Scalar Multiplication: αv = (αv₁, αv₂, ..., αvₙ)
   - Used in: Learning rates, regularization
   
3. Magnitude (L2 norm): ||v|| = √(v₁² + v₂² + ... + vₙ²)
   - Used in: Distance calculations, normalization
"""

# ML Example: Feature Scaling and Normalization
print("Example: Feature Scaling in ML")

# Original features (different scales)
features = np.array([
    [25, 50000, 16],      # age, income, education
    [35, 75000, 18],
    [28, 45000, 14],
    [42, 90000, 20]
])

print("Original features:")
print(features)
print(f"Feature means: {np.mean(features, axis=0)}")
print(f"Feature std: {np.std(features, axis=0)}")

# Z-score normalization (mean=0, std=1)
mean_vec = np.mean(features, axis=0)
std_vec = np.std(features, axis=0)
normalized_features = (features - mean_vec) / std_vec

print("\nNormalized features:")
print(normalized_features)
print(f"New means: {np.mean(normalized_features, axis=0)}")
print(f"New std: {np.std(normalized_features, axis=0)}")

# Min-Max scaling (range [0,1])
min_vec = np.min(features, axis=0)
max_vec = np.max(features, axis=0)
scaled_features = (features - min_vec) / (max_vec - min_vec)

print("\nMin-Max scaled features:")
print(scaled_features)

# 1.3 Dot Product - The Heart of ML
print("\n1.3 Dot Product - The Heart of ML")
print("-" * 35)

"""
THEORY:
The dot product is the most important operation in ML:
a · b = Σᵢ aᵢbᵢ = |a| |b| cos(θ)

Geometric Interpretation:
- Measures similarity between vectors
- Projects one vector onto another
- θ = 0°: vectors point same direction (max similarity)
- θ = 90°: vectors are orthogonal (no similarity)
- θ = 180°: vectors point opposite directions (max dissimilarity)

ML Applications:
- Neural networks: w · x + b
- SVM: decision boundaries
- Cosine similarity: text analysis
- Attention mechanisms: query · key
"""

# Similarity Analysis Example
def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Document vectors (word counts)
doc1 = np.array([2, 1, 0, 1, 0])  # "machine learning is great"
doc2 = np.array([1, 2, 1, 0, 0])  # "learning machine algorithms"
doc3 = np.array([0, 0, 3, 2, 1])  # "deep neural networks"

documents = [doc1, doc2, doc3]
doc_names = ['Doc 1', 'Doc 2', 'Doc 3']

print("Document Similarity Analysis:")
print("Vocabulary: [machine, learning, deep, neural, networks]")
for i, (doc, name) in enumerate(zip(documents, doc_names)):
    print(f"{name}: {doc}")

# Calculate all pairwise similarities
similarities = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        similarities[i, j] = cosine_similarity(documents[i], documents[j])

print("\nCosine Similarity Matrix:")
print(similarities)

# Visualize similarity matrix
plt.figure(figsize=(8, 6))
sns.heatmap(similarities, annot=True, cmap='coolwarm', center=0,
            xticklabels=doc_names, yticklabels=doc_names)
plt.title('Document Similarity Matrix')
plt.show()

# Neural Network Example: Simple Perceptron
print("\nExample: Simple Perceptron (Linear Classifier)")

def perceptron_predict(x, weights, bias):
    """Simple perceptron prediction"""
    return np.dot(x, weights) + bias

# Sample 2D data
X = np.array([[1, 2], [2, 3], [3, 1], [1, 1], [2, 1], [3, 3]])
y = np.array([1, 1, 1, -1, -1, -1])  # Binary classification

# Initialize weights
weights = np.array([0.5, -0.3])
bias = 0.1

predictions = []
for x in X:
    pred = perceptron_predict(x, weights, bias)
    predictions.append(pred)
    print(f"Input: {x}, w·x + b = {pred:.3f}, Class: {1 if pred > 0 else -1}")

# Visualize decision boundary
plt.figure(figsize=(10, 6))
colors = ['red' if label == -1 else 'blue' for label in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.7)

# Plot decision boundary
x_range = np.linspace(0, 4, 100)
y_boundary = -(weights[0] * x_range + bias) / weights[1]
plt.plot(x_range, y_boundary, 'k--', linewidth=2, label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# CHAPTER 2: MATRICES - DATA TRANSFORMATIONS
# =============================================================================

print("\n\n📚 CHAPTER 2: MATRICES - DATA TRANSFORMATIONS")
print("=" * 50)

# 2.1 Matrix Theory and ML Context
print("\n2.1 Matrix Theory and ML Context")
print("-" * 32)

"""
THEORY:
A matrix is a rectangular array of numbers: A ∈ ℝᵐˣⁿ

In Machine Learning:
- Data Matrix X: rows are samples, columns are features
- Weight Matrix W: transformation parameters
- Covariance Matrix Σ: feature relationships
- Transformation Matrix: linear mappings

Key Properties:
- Transpose: Aᵀ flips rows and columns
- Inverse: A⁻¹ exists if A is square and non-singular
- Determinant: det(A) measures volume scaling
- Rank: number of linearly independent rows/columns
"""

# ML Data Matrix Example
print("Example: ML Data Matrix")

# Create sample dataset
n_samples, n_features = 100, 4
X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                          n_redundant=0, n_informative=4, 
                          n_clusters_per_class=1, random_state=42)

print(f"Data matrix X shape: {X.shape}")
print(f"First 5 samples:")
print(X[:5])
print(f"\nTarget vector y shape: {y.shape}")
print(f"First 10 labels: {y[:10]}")

# Matrix properties
print(f"\nMatrix Properties:")
print(f"Mean of each feature: {np.mean(X, axis=0)}")
print(f"Std of each feature: {np.std(X, axis=0)}")
print(f"Matrix rank: {np.linalg.matrix_rank(X)}")

# 2.2 Matrix Operations in ML
print("\n2.2 Matrix Operations in ML")
print("-" * 28)

"""
THEORY:
Essential matrix operations for ML:

1. Matrix Multiplication: C = AB
   - Cᵢⱼ = Σₖ AᵢₖBₖⱼ
   - Used in: neural networks, transformations

2. Element-wise operations: A ⊙ B
   - Used in: activation functions, masking

3. Broadcasting: operations between different sized arrays
   - Used in: adding bias, normalization
"""

# Neural Network Forward Pass Example
print("Example: Neural Network Forward Pass")

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

# Network architecture: 4 → 8 → 4 → 1
input_size = 4
hidden1_size = 8
hidden2_size = 4
output_size = 1

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden1_size) * 0.5
b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(hidden1_size, hidden2_size) * 0.5
b2 = np.zeros((1, hidden2_size))

W3 = np.random.randn(hidden2_size, output_size) * 0.5
b3 = np.zeros((1, output_size))

# Forward pass for a batch
batch_size = 5
X_batch = X[:batch_size]

print(f"Input shape: {X_batch.shape}")

# Layer 1
z1 = X_batch @ W1 + b1  # Matrix multiplication + broadcasting
a1 = relu(z1)
print(f"Hidden layer 1 output shape: {a1.shape}")

# Layer 2
z2 = a1 @ W2 + b2
a2 = relu(z2)
print(f"Hidden layer 2 output shape: {a2.shape}")

# Output layer
z3 = a2 @ W3 + b3
output = sigmoid(z3)
print(f"Output shape: {output.shape}")
print(f"Predictions: {output.flatten()}")

# 2.3 Special Matrices in ML
print("\n2.3 Special Matrices in ML")
print("-" * 27)

"""
THEORY:
Special matrices with important ML applications:

1. Identity Matrix I: Iᵢⱼ = 1 if i=j, 0 otherwise
   - Neutral element for multiplication
   - Used in regularization

2. Diagonal Matrix: Non-zero elements only on main diagonal
   - Efficient computation
   - Scaling transformations

3. Symmetric Matrix: A = Aᵀ
   - Covariance matrices
   - Kernel matrices

4. Orthogonal Matrix: QᵀQ = I
   - Preserves distances
   - Rotations and reflections
"""

# Covariance Matrix Example
print("Example: Covariance Matrix Analysis")

# Calculate covariance matrix
cov_matrix = np.cov(X.T)  # Transpose because we want feature covariance
print(f"Covariance matrix shape: {cov_matrix.shape}")
print(f"Covariance matrix:\n{cov_matrix}")

# Properties of covariance matrix
print(f"\nCovariance matrix properties:")
print(f"Is symmetric: {np.allclose(cov_matrix, cov_matrix.T)}")
print(f"Diagonal elements (variances): {np.diag(cov_matrix)}")

# Visualize covariance matrix
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Covariance Matrix')

plt.subplot(1, 2, 2)
correlation_matrix = np.corrcoef(X.T)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Correlation Matrix')

plt.tight_layout()
plt.show()

# =============================================================================
# CHAPTER 3: EIGENVALUES AND EIGENVECTORS - UNDERSTANDING DATA STRUCTURE
# =============================================================================

print("\n\n📚 CHAPTER 3: EIGENVALUES AND EIGENVECTORS")
print("=" * 50)

# 3.1 Theory and Intuition
print("\n3.1 Theory and Intuition")
print("-" * 24)

"""
THEORY:
For a square matrix A, vector v is an eigenvector with eigenvalue λ if:
Av = λv

Geometric Interpretation:
- Eigenvectors are directions that don't change under transformation A
- Eigenvalues are scaling factors along those directions
- They reveal the "natural" directions of a matrix

ML Applications:
- Principal Component Analysis (PCA)
- Spectral clustering
- Markov chains (PageRank)
- Stability analysis in deep learning
"""

# Simple 2D Example
print("Example: 2D Transformation Matrix")

# Create a simple transformation matrix
A = np.array([[3, 1],
              [1, 3]])

# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Matrix A:\n{A}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify the eigenvalue equation
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    λ = eigenvalues[i]
    Av = A @ v
    λv = λ * v
    print(f"\nEigenvector {i+1}: {v}")
    print(f"A @ v = {Av}")
    print(f"λ * v = {λv}")
    print(f"Verification: {np.allclose(Av, λv)}")

# Visualization
plt.figure(figsize=(12, 5))

# Original and transformed vectors
test_vectors = np.array([[1, 0], [0, 1], [1, 1], [-1, 1]])
colors = ['red', 'blue', 'green', 'purple']

plt.subplot(1, 2, 1)
plt.title('Original Vectors')
for i, (vec, color) in enumerate(zip(test_vectors, colors)):
    plt.arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.1, 
              fc=color, ec=color, label=f'v{i+1}')
    
# Plot eigenvectors
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    plt.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1, 
              fc='black', ec='black', linestyle='--', alpha=0.7)
    plt.text(v[0]*1.2, v[1]*1.2, f'e{i+1}', fontsize=12)

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.title('Transformed Vectors (A * v)')
for i, (vec, color) in enumerate(zip(test_vectors, colors)):
    transformed = A @ vec
    plt.arrow(0, 0, transformed[0], transformed[1], head_width=0.1, head_length=0.1, 
              fc=color, ec=color, label=f'A*v{i+1}')

# Plot transformed eigenvectors
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    transformed_eigen = A @ v
    plt.arrow(0, 0, transformed_eigen[0], transformed_eigen[1], 
              head_width=0.1, head_length=0.1, fc='black', ec='black', 
              linestyle='--', alpha=0.7)
    plt.text(transformed_eigen[0]*1.1, transformed_eigen[1]*1.1, f'A*e{i+1}', fontsize=12)

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.show()

# 3.2 Principal Component Analysis (PCA)
print("\n3.2 Principal Component Analysis (PCA)")
print("-" * 37)

"""
THEORY:
PCA finds the directions of maximum variance in data:
1. Compute covariance matrix C
2. Find eigenvectors of C (principal components)
3. Project data onto top-k eigenvectors

Mathematical Steps:
1. Center data: X_centered = X - mean(X)
2. Covariance: C = (X_centered^T @ X_centered) / (n-1)
3. Eigendecomposition: C = V @ Λ @ V^T
4. Transform: X_pca = X_centered @ V[:, :k]
"""

# PCA Implementation from Scratch
print("Example: PCA Implementation from Scratch")

def pca_from_scratch(X, n_components=2):
    """Implement PCA from scratch"""
    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Step 4: Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Select top components
    principal_components = eigenvectors[:, :n_components]
    
    # Step 6: Transform data
    X_pca = X_centered @ principal_components
    
    return X_pca, principal_components, eigenvalues

# Generate sample data
np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
data_2d = np.random.multivariate_normal(mean, cov, 200)

# Apply PCA
X_pca, components, eigenvals = pca_from_scratch(data_2d, n_components=2)

print(f"Original data shape: {data_2d.shape}")
print(f"Principal components:\n{components}")
print(f"Eigenvalues: {eigenvals}")
print(f"Explained variance ratio: {eigenvals / np.sum(eigenvals)}")

# Visualization
plt.figure(figsize=(15, 5))

# Original data
plt.subplot(1, 3, 1)
plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Data with principal components
plt.subplot(1, 3, 2)
plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6)

# Plot principal components
mean_point = np.mean(data_2d, axis=0)
for i, (component, eigenval) in enumerate(zip(components.T, eigenvals)):
    # Scale by eigenvalue for visualization
    direction = component * np.sqrt(eigenval) * 2
    plt.arrow(mean_point[0], mean_point[1], direction[0], direction[1],
              head_width=0.1, head_length=0.1, fc=f'C{i+1}', ec=f'C{i+1}',
              linewidth=3, label=f'PC{i+1}')

plt.title('Data with Principal Components')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Transformed data
plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.title('PCA Transformed Data')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.show()

# Compare with sklearn PCA
from sklearn.decomposition import PCA

sklearn_pca = PCA(n_components=2)
sklearn_result = sklearn_pca.fit_transform(data_2d)

print(f"\nComparison with sklearn PCA:")
print(f"Our implementation explained variance: {eigenvals / np.sum(eigenvals)}")
print(f"Sklearn explained variance: {sklearn_pca.explained_variance_ratio_}")
print(f"Results match: {np.allclose(np.abs(X_pca), np.abs(sklearn_result))}")

# =============================================================================
# CHAPTER 4: MATRIX DECOMPOSITIONS - ADVANCED TECHNIQUES
# =============================================================================

print("\n\n📚 CHAPTER 4: MATRIX DECOMPOSITIONS")
print("=" * 40)

# 4.1 Singular Value Decomposition (SVD)
print("\n4.1 Singular Value Decomposition (SVD)")
print("-" * 39)

"""
THEORY:
SVD decomposes any matrix A into: A = UΣV^T

Where:
- U: left singular vectors (orthogonal)
- Σ: diagonal matrix of singular values
- V: right singular vectors (orthogonal)

Properties:
- Works for any matrix (not just square)
- Singular values are non-negative
- Related to eigenvalues: σᵢ = √λᵢ

ML Applications:
- Dimensionality reduction
- Recommender systems
- Data compression
- Principal Component Analysis
- Latent Semantic Analysis
"""

# SVD Example: Image Compression
print("Example: Image Compression using SVD")

# Create a simple synthetic image
def create_sample_image():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1 * (X**2 + Y**2))
    return Z

image = create_sample_image()
print(f"Original image shape: {image.shape}")

# Apply SVD
U, sigma, Vt = np.linalg.svd(image, full_matrices=False)

print(f"U shape: {U.shape}")
print(f"Sigma shape: {sigma.shape}")
print(f"V^T shape: {Vt.shape}")

# Reconstruct with different numbers of components
ranks = [1, 5, 10, 25, 50]
reconstructions = []

for rank in ranks:
    # Reconstruct using only first 'rank' components
    reconstruction = U[:, :rank] @ np.diag(sigma[:rank]) @ Vt[:rank, :]
    reconstructions.append(reconstruction)
    
    # Calculate compression ratio
    original_size = image.shape[0] * image.shape[1]
    compressed_size = rank * (U.shape[0] + Vt.shape[1]) + rank
    compression_ratio = compressed_size / original_size
    
    print(f"Rank {rank}: Compression ratio = {compression_ratio:.3f}")

# Visualize reconstructions
plt.figure(figsize=(18, 12))

# Original image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='viridis')
plt.title('Original Image')
plt.colorbar()

# Reconstructions
for i, (rank, reconstruction) in enumerate(zip(ranks, reconstructions)):
    plt.subplot(2, 3, i+2)
    plt.imshow(reconstruction, cmap='viridis')
    plt.title(f'Rank {rank} Approximation')
    plt.colorbar()

plt.tight_layout()
plt.show()

# Plot singular values
plt.figure(figsize=(10, 6))
plt.plot(sigma[:50], 'bo-', markersize=4)
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value')
plt.title('Singular Values (Energy Distribution)')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()

# 4.2 QR Decomposition
print("\n4.2 QR Decomposition")
print("-" * 20)

"""
THEORY:
QR decomposition factors A into: A = QR

Where:
- Q: orthogonal matrix (Q^T Q = I)
- R: upper triangular matrix

Applications:
- Solving linear systems
- Least squares problems
- Gram-Schmidt orthogonalization
- Numerical stability in computations
"""

# QR Example: Solving Linear Regression
print("Example: Linear Regression using QR Decomposition")

# Generate regression data
np.random.seed(42)
n_samples, n_features = 100, 3
X = np.random.randn(n_samples, n_features)
true_weights = np.array([1.5, -2.0, 0.8])
noise = 0.1 * np.random.randn(n_samples)
y = X @ true_weights + noise

# Add intercept term
X_with_intercept = np.column_stack([np.ones(n_samples), X])

print(f"Design matrix shape: {X_with_intercept.shape}")
print(f"True weights (with intercept): [0, {true_weights[0]}, {true_weights[1]}, {true_weights[2]}]")

# Method 1: Normal equation (can be unstable)
def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Method 2: QR decomposition (more stable)
def qr_solve(X, y):
    Q, R = np.linalg.qr(X)
    return np.linalg.solve(R, Q.T @ y)

# Method 3: SVD (most stable)
def svd_solve(X, y):
    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ (U.T @ y / sigma)

# Compare methods
weights_normal = normal_