# 01_gradient_based_optimization.ipynb
## Gradient-Based Optimization: Theory and Implementation

### Table of Contents
1. [Mathematical Foundations](#mathematical-foundations)
2. [NumPy Implementation](#numpy-implementation)
3. [Visualizations](#visualizations)
4. [ML Use Case: Linear Regression](#ml-use-case-linear-regression)
5. [Exercises](#exercises)

---

## Mathematical Foundations

### 1.1 Gradients and Partial Derivatives

The **gradient** of a function represents the direction of steepest increase. For a function $f(x_1, x_2, ..., x_n)$, the gradient is:

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right)$$

#### Key Properties:
- **Direction**: Points toward steepest increase
- **Magnitude**: Rate of change in that direction
- **Optimization**: We move opposite to gradient (negative gradient) to minimize

#### Example: Quadratic Function
For $f(x, y) = x^2 + y^2$:
- $\frac{\partial f}{\partial x} = 2x$
- $\frac{\partial f}{\partial y} = 2y$
- $\nabla f = (2x, 2y)$

### 1.2 Gradient Descent Algorithm

The gradient descent update rule is:

$$\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)$$

Where:
- $\theta_t$ = parameters at iteration $t$
- $\alpha$ = learning rate (step size)
- $\nabla f(\theta_t)$ = gradient at current parameters

#### Variants:
1. **Vanilla (Batch) Gradient Descent**: Uses entire dataset
2. **Stochastic Gradient Descent (SGD)**: Uses one sample at a time
3. **Mini-batch Gradient Descent**: Uses small batches

---

## NumPy Implementation

### 2.1 Vanilla Gradient Descent

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def vanilla_gradient_descent(gradient_func, initial_params, learning_rate=0.01, 
                           max_iterations=1000, tolerance=1e-6):
    """
    Implement vanilla gradient descent algorithm
    
    Parameters:
    - gradient_func: Function that returns gradient at given parameters
    - initial_params: Starting parameters
    - learning_rate: Step size (alpha)
    - max_iterations: Maximum number of iterations
    - tolerance: Convergence threshold
    
    Returns:
    - params_history: History of parameters
    - cost_history: History of cost values
    """
    params = initial_params.copy()
    params_history = [params.copy()]
    cost_history = []
    
    for i in range(max_iterations):
        # Calculate gradient
        gradient = gradient_func(params)
        
        # Update parameters
        params = params - learning_rate * gradient
        
        # Store history
        params_history.append(params.copy())
        
        # Check convergence
        if np.linalg.norm(gradient) < tolerance:
            print(f"Converged after {i+1} iterations")
            break
    
    return np.array(params_history), params

# Example usage with quadratic function
def quadratic_gradient(params):
    """Gradient of f(x,y) = x^2 + y^2"""
    return 2 * params

# Run optimization
initial = np.array([4.0, 3.0])
history, final_params = vanilla_gradient_descent(
    quadratic_gradient, initial, learning_rate=0.1
)

print(f"Final parameters: {final_params}")
print(f"Number of iterations: {len(history)-1}")
```

### 2.2 Stochastic Gradient Descent

```python
def stochastic_gradient_descent(X, y, initial_params, learning_rate=0.01, 
                               max_iterations=1000, random_seed=42):
    """
    Implement stochastic gradient descent for linear regression
    
    Parameters:
    - X: Feature matrix (n_samples, n_features)
    - y: Target values (n_samples,)
    - initial_params: Starting parameters [bias, weights...]
    - learning_rate: Step size
    - max_iterations: Maximum iterations
    - random_seed: For reproducibility
    
    Returns:
    - params_history: Parameter evolution
    - cost_history: Cost function evolution
    """
    np.random.seed(random_seed)
    
    n_samples, n_features = X.shape
    params = initial_params.copy()
    
    params_history = [params.copy()]
    cost_history = []
    
    # Add bias column to X
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    
    for iteration in range(max_iterations):
        # Shuffle data for each epoch
        indices = np.random.permutation(n_samples)
        
        for i in indices:
            # Select single sample
            xi = X_with_bias[i:i+1]  # Shape: (1, n_features+1)
            yi = y[i:i+1]           # Shape: (1,)
            
            # Forward pass
            prediction = xi @ params
            error = prediction - yi
            
            # Compute gradient for single sample
            gradient = xi.T @ error  # Shape: (n_features+1, 1)
            gradient = gradient.flatten()
            
            # Update parameters
            params = params - learning_rate * gradient
            
            # Store history (every 10 updates to reduce memory)
            if len(params_history) % 10 == 0:
                params_history.append(params.copy())
        
        # Calculate cost on full dataset
        predictions = X_with_bias @ params
        cost = np.mean((predictions - y) ** 2)
        cost_history.append(cost)
    
    return np.array(params_history), np.array(cost_history), params
```

### 2.3 Comparison Function

```python
def compare_optimizers(X, y, initial_params, learning_rates, max_iter=1000):
    """Compare vanilla GD vs SGD performance"""
    
    results = {}
    
    # Vanilla GD
    def mse_gradient(params):
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        predictions = X_with_bias @ params
        errors = predictions - y
        return (2/len(X)) * X_with_bias.T @ errors
    
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        
        # Vanilla GD
        history_vanilla, _ = vanilla_gradient_descent(
            mse_gradient, initial_params, lr, max_iter
        )
        
        # SGD
        history_sgd, cost_sgd, _ = stochastic_gradient_descent(
            X, y, initial_params, lr, max_iter
        )
        
        results[lr] = {
            'vanilla': history_vanilla,
            'sgd': history_sgd,
            'sgd_cost': cost_sgd
        }
    
    return results
```

---

## Visualizations

### 3.1 Loss Surface Visualization

```python
def plot_loss_surface():
    """Visualize loss surface for quadratic function"""
    
    # Create grid
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Quadratic function
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Loss Surface: f(x,y) = x² + y²')
    
    # Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X, Y, Z, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Contour Plot')
    ax2.grid(True, alpha=0.3)
    
    # Gradient field
    ax3 = fig.add_subplot(133)
    # Sample points for gradient field
    x_grad = np.linspace(-4, 4, 15)
    y_grad = np.linspace(-4, 4, 15)
    X_grad, Y_grad = np.meshgrid(x_grad, y_grad)
    U = 2 * X_grad  # ∂f/∂x
    V = 2 * Y_grad  # ∂f/∂y
    
    ax3.quiver(X_grad, Y_grad, U, V, alpha=0.7)
    ax3.contour(X, Y, Z, levels=10, alpha=0.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Gradient Field')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Execute the visualization
plot_loss_surface()
```

### 3.2 Learning Rate Effect Visualization

```python
def plot_learning_rate_effect():
    """Demonstrate effect of different learning rates"""
    
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    colors = ['blue', 'green', 'orange', 'red']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    axes = [ax1, ax2, ax3, ax4]
    
    # Create contour background
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    for i, (lr, color, ax) in enumerate(zip(learning_rates, colors, axes)):
        # Run optimization
        initial = np.array([4.0, 3.0])
        history, _ = vanilla_gradient_descent(
            quadratic_gradient, initial, learning_rate=lr, max_iterations=50
        )
        
        # Plot contours
        ax.contour(X, Y, Z, levels=15, alpha=0.5, colors='gray')
        
        # Plot optimization path
        ax.plot(history[:, 0], history[:, 1], 'o-', 
                color=color, markersize=4, linewidth=2, alpha=0.8)
        ax.plot(history[0, 0], history[0, 1], 'go', markersize=8, label='Start')
        ax.plot(history[-1, 0], history[-1, 1], 'ro', markersize=8, label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Learning Rate = {lr}\nIterations: {len(history)-1}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    
    plt.suptitle('Effect of Learning Rate on Convergence', fontsize=16)
    plt.tight_layout()
    plt.show()

# Execute the visualization
plot_learning_rate_effect()
```

### 3.3 Convergence Plot

```python
def plot_convergence():
    """Plot convergence behavior for different algorithms"""
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 1)
    true_params = np.array([2.0, 3.0])  # [bias, weight]
    y = true_params[0] + true_params[1] * X.flatten() + 0.1 * np.random.randn(100)
    
    initial_params = np.array([0.0, 0.0])
    learning_rate = 0.01
    
    # Run optimizers
    results = compare_optimizers(X, y, initial_params, [learning_rate], max_iter=100)
    
    # Calculate costs for vanilla GD
    def calculate_cost(params_history):
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        costs = []
        for params in params_history:
            predictions = X_with_bias @ params
            cost = np.mean((predictions - y) ** 2)
            costs.append(cost)
        return costs
    
    vanilla_costs = calculate_cost(results[learning_rate]['vanilla'])
    sgd_costs = results[learning_rate]['sgd_cost']
    
    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cost vs iterations
    ax1.plot(range(len(vanilla_costs)), vanilla_costs, 'b-', 
             label='Vanilla GD', linewidth=2)
    ax1.plot(range(len(sgd_costs)), sgd_costs, 'r-', 
             label='SGD', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Parameter evolution
    vanilla_history = results[learning_rate]['vanilla']
    sgd_history = results[learning_rate]['sgd']
    
    ax2.plot(vanilla_history[:, 0], label='Vanilla GD - Bias', linewidth=2)
    ax2.plot(vanilla_history[:, 1], label='Vanilla GD - Weight', linewidth=2)
    ax2.axhline(y=true_params[0], color='blue', linestyle='--', alpha=0.5, label='True Bias')
    ax2.axhline(y=true_params[1], color='orange', linestyle='--', alpha=0.5, label='True Weight')
    
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Execute the visualization
plot_convergence()
```

---

## ML Use Case: Linear Regression

### 4.1 Linear Regression from Scratch

```python
class LinearRegressionGD:
    """Linear Regression using Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.params = None
        self.cost_history = []
        self.params_history = []
    
    def fit(self, X, y, method='vanilla'):
        """
        Fit linear regression model
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - y: Target values (n_samples,)
        - method: 'vanilla' or 'stochastic'
        """
        # Initialize parameters
        n_features = X.shape[1]
        self.params = np.zeros(n_features + 1)  # +1 for bias
        
        # Add bias column
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        
        if method == 'vanilla':
            self._fit_vanilla(X_with_bias, y)
        elif method == 'stochastic':
            self._fit_stochastic(X_with_bias, y)
        else:
            raise ValueError("Method must be 'vanilla' or 'stochastic'")
    
    def _fit_vanilla(self, X_with_bias, y):
        """Vanilla gradient descent implementation"""
        for iteration in range(self.max_iterations):
            # Forward pass
            predictions = X_with_bias @ self.params
            errors = predictions - y
            
            # Calculate cost
            cost = np.mean(errors ** 2)
            self.cost_history.append(cost)
            
            # Calculate gradient
            gradient = (2/len(X_with_bias)) * X_with_bias.T @ errors
            
            # Update parameters
            self.params = self.params - self.learning_rate * gradient
            self.params_history.append(self.params.copy())
            
            # Check convergence
            if np.linalg.norm(gradient) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
    
    def _fit_stochastic(self, X_with_bias, y):
        """Stochastic gradient descent implementation"""
        n_samples = len(X_with_bias)
        
        for iteration in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                # Single sample
                xi = X_with_bias[i:i+1]
                yi = y[i:i+1]
                
                # Forward pass
                prediction = xi @ self.params
                error = prediction - yi
                
                # Gradient for single sample
                gradient = xi.T @ error
                gradient = gradient.flatten()
                
                # Update parameters
                self.params = self.params - self.learning_rate * gradient
            
            # Calculate cost on full dataset
            predictions = X_with_bias @ self.params
            cost = np.mean((predictions - y) ** 2)
            self.cost_history.append(cost)
            self.params_history.append(self.params.copy())
    
    def predict(self, X):
        """Make predictions"""
        if self.params is None:
            raise ValueError("Model not fitted yet!")
        
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        return X_with_bias @ self.params
    
    def score(self, X, y):
        """Calculate R-squared score"""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
```

### 4.2 Complete Example

```python
# Generate synthetic dataset
np.random.seed(42)
n_samples = 200
n_features = 2

# Create features
X = np.random.randn(n_samples, n_features)
X[:, 1] = X[:, 1] * 2  # Scale second feature

# True parameters
true_bias = 3.0
true_weights = np.array([2.5, -1.8])

# Generate target with noise
y = true_bias + X @ true_weights + 0.3 * np.random.randn(n_samples)

# Split data
split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train models
model_vanilla = LinearRegressionGD(learning_rate=0.01, max_iterations=1000)
model_sgd = LinearRegressionGD(learning_rate=0.001, max_iterations=100)

print("Training Vanilla GD...")
model_vanilla.fit(X_train, y_train, method='vanilla')

print("\nTraining SGD...")
model_sgd.fit(X_train, y_train, method='stochastic')

# Evaluate models
train_score_vanilla = model_vanilla.score(X_train, y_train)
test_score_vanilla = model_vanilla.score(X_test, y_test)

train_score_sgd = model_sgd.score(X_train, y_train)
test_score_sgd = model_sgd.score(X_test, y_test)

print(f"\nResults:")
print(f"True parameters: Bias={true_bias:.3f}, Weights={true_weights}")
print(f"Vanilla GD: Bias={model_vanilla.params[0]:.3f}, Weights={model_vanilla.params[1:]}")
print(f"SGD: Bias={model_sgd.params[0]:.3f}, Weights={model_sgd.params[1:]}")
print(f"\nVanilla GD - Train R²: {train_score_vanilla:.4f}, Test R²: {test_score_vanilla:.4f}")
print(f"SGD - Train R²: {train_score_sgd:.4f}, Test R²: {test_score_sgd:.4f}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Cost curves
ax1.plot(model_vanilla.cost_history, label='Vanilla GD', linewidth=2)
ax1.plot(model_sgd.cost_history, label='SGD', linewidth=2, alpha=0.7)
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('Training Cost Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Predictions vs actual
predictions_vanilla = model_vanilla.predict(X_test)
predictions_sgd = model_sgd.predict(X_test)

ax2.scatter(y_test, predictions_vanilla, alpha=0.6, label='Vanilla GD', s=30)
ax2.scatter(y_test, predictions_sgd, alpha=0.6, label='SGD', s=30)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Values')
ax2.set_ylabel('Predictions')
ax2.set_title('Predictions vs Actual (Test Set)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Exercises

### Exercise 1: Custom Optimization Function
Implement gradient descent for the Rosenbrock function:
$$f(x,y) = 100(y-x^2)^2 + (1-x)^2$$

**Tasks:**
1. Derive the gradient analytically
2. Implement the optimization
3. Visualize the optimization path
4. Experiment with different learning rates

```python
def rosenbrock(params):
    """Rosenbrock function"""
    x, y = params
    return 100 * (y - x**2)**2 + (1 - x)**2

def rosenbrock_gradient(params):
    """Gradient of Rosenbrock function"""
    x, y = params
    dx = -400 * x * (y - x**2) - 2 * (1 - x)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# Your implementation here
```

### Exercise 2: Momentum and Adaptive Learning Rates
Extend the basic gradient descent with:
1. Momentum term
2. Adaptive learning rate (e.g., decay)
3. Compare performance with vanilla GD

### Exercise 3: Regularized Linear Regression
Implement Ridge regression (L2 regularization) using gradient descent:
$$J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\sum_{j=1}^{n}\theta_j^2$$

### Exercise 4: Multi-class Classification
Implement logistic regression for multi-class classification using:
1. One-vs-rest approach
2. Softmax regression
3. Compare with scikit-learn implementations

---

## Summary

This notebook covered:

1. **Mathematical foundations** of gradients and optimization
2. **NumPy implementations** of vanilla and stochastic gradient descent
3. **Visualizations** showing loss surfaces, learning rate effects, and convergence
4. **Practical ML application** with linear regression from scratch

### Key Takeaways:
- Gradient descent is fundamental to most ML algorithms
- Learning rate significantly affects convergence behavior
- Stochastic variants trade accuracy for speed
- Visualization helps understand optimization dynamics
- Implementation from scratch deepens understanding

### Next Steps:
- Explore advanced optimizers (Adam, RMSprop)
- Study second-order methods (Newton's method)
- Apply to neural networks
- Investigate constrained optimization