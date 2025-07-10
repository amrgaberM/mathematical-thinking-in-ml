# 03 - Regularization Techniques

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Bias-Variance Tradeoff](#bias-variance-tradeoff)
4. [Implementation from Scratch](#implementation-from-scratch)
5. [Visualizing Regularization Effects](#visualizing-regularization-effects)
6. [Ridge vs Lasso with scikit-learn](#ridge-vs-lasso-with-scikit-learn)
7. [Conclusion](#conclusion)

---

## Introduction

Regularization is a crucial technique in machine learning that helps prevent overfitting by adding a penalty term to the loss function. This notebook explores L1 and L2 regularization, their mathematical foundations, and practical implementations.

**Learning Objectives:**
- Understand L1 and L2 penalty terms mathematically
- Grasp the bias-variance tradeoff in regularization
- Implement regularized linear regression from scratch
- Visualize the effects of regularization strength
- Compare Ridge and Lasso regression using scikit-learn

---

## Mathematical Foundation

### 📘 L1 and L2 Penalty Terms

#### Standard Linear Regression Loss
The standard mean squared error (MSE) loss for linear regression is:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Where:
- $h_\theta(x) = \theta^T x$ is our hypothesis
- $m$ is the number of training examples
- $\theta$ are the model parameters

#### L2 Regularization (Ridge)
L2 regularization adds a penalty proportional to the sum of squares of parameters:

$$J_{Ridge}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2$$

**Key Properties:**
- Penalty term: $\lambda \|\theta\|_2^2 = \lambda \sum_{j=1}^{n} \theta_j^2$
- Encourages small parameter values
- Differentiable everywhere
- Tends to shrink coefficients towards zero but rarely makes them exactly zero

#### L1 Regularization (Lasso)
L1 regularization adds a penalty proportional to the sum of absolute values of parameters:

$$J_{Lasso}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} |\theta_j|$$

**Key Properties:**
- Penalty term: $\lambda \|\theta\|_1 = \lambda \sum_{j=1}^{n} |\theta_j|$
- Encourages sparsity (feature selection)
- Not differentiable at zero
- Can drive coefficients to exactly zero

#### Elastic Net
Combines both L1 and L2 penalties:

$$J_{Elastic}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda_1 \sum_{j=1}^{n} |\theta_j| + \lambda_2 \sum_{j=1}^{n} \theta_j^2$$

---

## Bias-Variance Tradeoff

### 🧠 Understanding the Tradeoff

The bias-variance tradeoff is fundamental to understanding regularization's effectiveness.

#### Mathematical Decomposition
For any learning algorithm, the expected prediction error can be decomposed as:

$$E[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

Where:
- **Bias**: $\text{Bias}[\hat{f}(x)] = E[\hat{f}(x)] - f(x)$
- **Variance**: $\text{Var}[\hat{f}(x)] = E[\hat{f}(x)^2] - E[\hat{f}(x)]^2$
- **Irreducible Error**: $\sigma^2$ (noise in the data)

#### Effect of Regularization

| Regularization Strength | Bias | Variance | Total Error |
|-------------------------|------|----------|-------------|
| None (λ = 0) | Low | High | High (overfitting) |
| Low (λ small) | Low-Medium | Medium-High | Medium |
| Optimal (λ*) | Medium | Medium | **Minimum** |
| High (λ large) | High | Low | High (underfitting) |

**Key Insights:**
- Regularization increases bias but decreases variance
- The goal is to find the optimal λ that minimizes total error
- Cross-validation helps find this optimal balance

---

## Implementation from Scratch

### 💻 NumPy Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RegularizedLinearRegression:
    """
    Linear Regression with L1 (Lasso) and L2 (Ridge) regularization
    """
    
    def __init__(self, reg_type='ridge', lambda_reg=0.01, learning_rate=0.01, n_iterations=1000):
        self.reg_type = reg_type.lower()
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.cost_history = []
        
    def _add_intercept(self, X):
        """Add bias column to the feature matrix"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _compute_cost(self, X, y, theta):
        """Compute the regularized cost function"""
        m = X.shape[0]
        
        # Basic MSE cost
        predictions = X.dot(theta)
        mse_cost = np.sum((predictions - y) ** 2) / (2 * m)
        
        # Regularization term (excluding bias term)
        reg_cost = 0
        if self.reg_type == 'ridge':
            reg_cost = self.lambda_reg * np.sum(theta[1:] ** 2)
        elif self.reg_type == 'lasso':
            reg_cost = self.lambda_reg * np.sum(np.abs(theta[1:]))
        
        return mse_cost + reg_cost
    
    def _compute_gradient(self, X, y, theta):
        """Compute the regularized gradient"""
        m = X.shape[0]
        
        # Basic gradient
        predictions = X.dot(theta)
        gradient = X.T.dot(predictions - y) / m
        
        # Add regularization to gradient (excluding bias term)
        if self.reg_type == 'ridge':
            gradient[1:] += 2 * self.lambda_reg * theta[1:]
        elif self.reg_type == 'lasso':
            gradient[1:] += self.lambda_reg * np.sign(theta[1:])
        
        return gradient
    
    def fit(self, X, y):
        """Train the regularized linear regression model"""
        # Add intercept term
        X_with_intercept = self._add_intercept(X)
        
        # Initialize parameters
        self.theta = np.random.normal(0, 0.01, X_with_intercept.shape[1])
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Compute cost and gradient
            cost = self._compute_cost(X_with_intercept, y, self.theta)
            gradient = self._compute_gradient(X_with_intercept, y, self.theta)
            
            # Update parameters
            self.theta -= self.learning_rate * gradient
            
            # Store cost
            self.cost_history.append(cost)
            
            # Print progress
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")
    
    def predict(self, X):
        """Make predictions on new data"""
        X_with_intercept = self._add_intercept(X)
        return X_with_intercept.dot(self.theta)
    
    def plot_cost_history(self):
        """Plot the cost function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title(f'Cost Function ({self.reg_type.title()} Regularization)')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()

# Generate synthetic dataset
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models with different regularization
models = {
    'No Regularization': RegularizedLinearRegression(reg_type='none', lambda_reg=0),
    'Ridge (λ=0.1)': RegularizedLinearRegression(reg_type='ridge', lambda_reg=0.1),
    'Lasso (λ=0.1)': RegularizedLinearRegression(reg_type='lasso', lambda_reg=0.1)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = np.mean((y_train - y_pred_train) ** 2)
    test_mse = np.mean((y_test - y_pred_test) ** 2)
    
    results[name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'parameters': model.theta.copy()
    }
    
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
```

---

## Visualizing Regularization Effects

### 📊 Effect of Regularization Strength

```python
# Function to analyze regularization strength
def analyze_regularization_strength():
    """Analyze the effect of different regularization strengths"""
    
    # Generate more complex dataset
    np.random.seed(42)
    X, y = make_regression(n_samples=50, n_features=10, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different lambda values
    lambda_values = np.logspace(-4, 2, 50)  # From 0.0001 to 100
    
    ridge_results = {'train_mse': [], 'test_mse': [], 'coefficients': []}
    lasso_results = {'train_mse': [], 'test_mse': [], 'coefficients': []}
    
    for lambda_reg in lambda_values:
        # Ridge Regression
        ridge_model = RegularizedLinearRegression(
            reg_type='ridge', 
            lambda_reg=lambda_reg, 
            learning_rate=0.01, 
            n_iterations=1000
        )
        ridge_model.fit(X_train_scaled, y_train)
        
        ridge_train_pred = ridge_model.predict(X_train_scaled)
        ridge_test_pred = ridge_model.predict(X_test_scaled)
        
        ridge_results['train_mse'].append(np.mean((y_train - ridge_train_pred) ** 2))
        ridge_results['test_mse'].append(np.mean((y_test - ridge_test_pred) ** 2))
        ridge_results['coefficients'].append(ridge_model.theta[1:].copy())  # Exclude bias
        
        # Lasso Regression
        lasso_model = RegularizedLinearRegression(
            reg_type='lasso', 
            lambda_reg=lambda_reg, 
            learning_rate=0.01, 
            n_iterations=1000
        )
        lasso_model.fit(X_train_scaled, y_train)
        
        lasso_train_pred = lasso_model.predict(X_train_scaled)
        lasso_test_pred = lasso_model.predict(X_test_scaled)
        
        lasso_results['train_mse'].append(np.mean((y_train - lasso_train_pred) ** 2))
        lasso_results['test_mse'].append(np.mean((y_test - lasso_test_pred) ** 2))
        lasso_results['coefficients'].append(lasso_model.theta[1:].copy())  # Exclude bias
    
    return lambda_values, ridge_results, lasso_results

# Run analysis
lambda_values, ridge_results, lasso_results = analyze_regularization_strength()

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Training and Test Error vs Lambda (Ridge)
axes[0, 0].semilogx(lambda_values, ridge_results['train_mse'], label='Train MSE', linewidth=2)
axes[0, 0].semilogx(lambda_values, ridge_results['test_mse'], label='Test MSE', linewidth=2)
axes[0, 0].set_xlabel('λ (Regularization Strength)')
axes[0, 0].set_ylabel('Mean Squared Error')
axes[0, 0].set_title('Ridge Regression: Bias-Variance Tradeoff')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Training and Test Error vs Lambda (Lasso)
axes[0, 1].semilogx(lambda_values, lasso_results['train_mse'], label='Train MSE', linewidth=2)
axes[0, 1].semilogx(lambda_values, lasso_results['test_mse'], label='Test MSE', linewidth=2)
axes[0, 1].set_xlabel('λ (Regularization Strength)')
axes[0, 1].set_ylabel('Mean Squared Error')
axes[0, 1].set_title('Lasso Regression: Bias-Variance Tradeoff')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Coefficient Paths (Ridge)
ridge_coefficients = np.array(ridge_results['coefficients'])
for i in range(ridge_coefficients.shape[1]):
    axes[1, 0].semilogx(lambda_values, ridge_coefficients[:, i], linewidth=2)
axes[1, 0].set_xlabel('λ (Regularization Strength)')
axes[1, 0].set_ylabel('Coefficient Value')
axes[1, 0].set_title('Ridge: Coefficient Shrinkage')
axes[1, 0].grid(True, alpha=0.3)

# 4. Coefficient Paths (Lasso)
lasso_coefficients = np.array(lasso_results['coefficients'])
for i in range(lasso_coefficients.shape[1]):
    axes[1, 1].semilogx(lambda_values, lasso_coefficients[:, i], linewidth=2)
axes[1, 1].set_xlabel('λ (Regularization Strength)')
axes[1, 1].set_ylabel('Coefficient Value')
axes[1, 1].set_title('Lasso: Coefficient Shrinkage & Selection')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print optimal lambda values
ridge_optimal_idx = np.argmin(ridge_results['test_mse'])
lasso_optimal_idx = np.argmin(lasso_results['test_mse'])

print(f"\nOptimal λ for Ridge: {lambda_values[ridge_optimal_idx]:.4f}")
print(f"Minimum Test MSE (Ridge): {ridge_results['test_mse'][ridge_optimal_idx]:.4f}")
print(f"\nOptimal λ for Lasso: {lambda_values[lasso_optimal_idx]:.4f}")
print(f"Minimum Test MSE (Lasso): {lasso_results['test_mse'][lasso_optimal_idx]:.4f}")
```

---

## Ridge vs Lasso with scikit-learn

### ⚙️ ML Use Case: Comparing Ridge and Lasso

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Generate a more realistic dataset
np.random.seed(42)
n_samples, n_features = 100, 20
X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                      n_informative=10, noise=10, random_state=42)

# Add some correlated features to make the problem more interesting
X_corr = X.copy()
for i in range(5):
    X_corr = np.column_stack([X_corr, X[:, i] + np.random.normal(0, 0.1, n_samples)])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_corr, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models and parameter grids
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}

param_grids = {
    'Ridge': {'alpha': np.logspace(-4, 2, 50)},
    'Lasso': {'alpha': np.logspace(-4, 2, 50)},
    'ElasticNet': {'alpha': np.logspace(-4, 2, 20), 'l1_ratio': [0.1, 0.5, 0.7, 0.9]}
}

# Perform grid search for each model
best_models = {}
for name, model in models.items():
    print(f"\nOptimizing {name}...")
    
    grid_search = GridSearchCV(
        model, 
        param_grids[name], 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    best_models[name] = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.4f}")

# Compare models
comparison_results = []

for name, model in best_models.items():
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Count non-zero coefficients
    non_zero_coefs = np.sum(np.abs(model.coef_) > 1e-5)
    
    comparison_results.append({
        'Model': name,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Non-zero Coefficients': non_zero_coefs,
        'Total Features': len(model.coef_)
    })

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_results)
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False, float_format='%.4f'))

# Visualize coefficient comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (name, model) in enumerate(best_models.items()):
    axes[i].bar(range(len(model.coef_)), model.coef_, alpha=0.7)
    axes[i].set_title(f'{name} Coefficients')
    axes[i].set_xlabel('Feature Index')
    axes[i].set_ylabel('Coefficient Value')
    axes[i].grid(True, alpha=0.3)
    
    # Add horizontal line at zero
    axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()

# Feature selection analysis
print("\n" + "="*80)
print("FEATURE SELECTION ANALYSIS")
print("="*80)

for name, model in best_models.items():
    print(f"\n{name}:")
    print(f"  Total features: {len(model.coef_)}")
    print(f"  Non-zero coefficients: {np.sum(np.abs(model.coef_) > 1e-5)}")
    print(f"  Sparsity: {(1 - np.sum(np.abs(model.coef_) > 1e-5) / len(model.coef_)) * 100:.1f}%")
    
    # Show top 5 most important features
    feature_importance = np.abs(model.coef_)
    top_features = np.argsort(feature_importance)[-5:][::-1]
    print(f"  Top 5 features: {top_features} with coefficients: {model.coef_[top_features]}")
```

### Cross-Validation Analysis

```python
# Perform detailed cross-validation analysis
def cross_validation_analysis():
    """Detailed cross-validation analysis comparing regularization methods"""
    
    # Different alpha values to test
    alphas = np.logspace(-4, 2, 30)
    
    # Store results
    ridge_scores = []
    lasso_scores = []
    
    for alpha in alphas:
        # Ridge
        ridge = Ridge(alpha=alpha)
        ridge_cv_scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5, 
                                         scoring='neg_mean_squared_error')
        ridge_scores.append(-ridge_cv_scores.mean())
        
        # Lasso
        lasso = Lasso(alpha=alpha, max_iter=2000)
        lasso_cv_scores = cross_val_score(lasso, X_train_scaled, y_train, cv=5, 
                                         scoring='neg_mean_squared_error')
        lasso_scores.append(-lasso_cv_scores.mean())
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(alphas, ridge_scores, 'b-', linewidth=2, label='Ridge')
    plt.semilogx(alphas, lasso_scores, 'r-', linewidth=2, label='Lasso')
    plt.xlabel('α (Regularization Strength)')
    plt.ylabel('Cross-Validation MSE')
    plt.title('Cross-Validation: Ridge vs Lasso')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Find optimal alphas
    ridge_optimal_alpha = alphas[np.argmin(ridge_scores)]
    lasso_optimal_alpha = alphas[np.argmin(lasso_scores)]
    
    plt.axvline(ridge_optimal_alpha, color='blue', linestyle='--', alpha=0.7, 
                label=f'Ridge optimal: {ridge_optimal_alpha:.4f}')
    plt.axvline(lasso_optimal_alpha, color='red', linestyle='--', alpha=0.7, 
                label=f'Lasso optimal: {lasso_optimal_alpha:.4f}')
    
    # Coefficient paths
    plt.subplot(1, 2, 2)
    ridge_coefs = []
    lasso_coefs = []
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        ridge_coefs.append(ridge.coef_)
        
        lasso = Lasso(alpha=alpha, max_iter=2000)
        lasso.fit(X_train_scaled, y_train)
        lasso_coefs.append(lasso.coef_)
    
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)
    
    # Plot a few coefficient paths
    for i in range(min(5, ridge_coefs.shape[1])):
        plt.semilogx(alphas, ridge_coefs[:, i], 'b-', alpha=0.6)
        plt.semilogx(alphas, lasso_coefs[:, i], 'r-', alpha=0.6)
    
    plt.xlabel('α (Regularization Strength)')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficient Paths')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return ridge_optimal_alpha, lasso_optimal_alpha

ridge_opt, lasso_opt = cross_validation_analysis()
print(f"\nOptimal α values from CV:")
print(f"Ridge: {ridge_opt:.4f}")
print(f"Lasso: {lasso_opt:.4f}")
```

---

## Conclusion

### Key Takeaways

1. **Mathematical Understanding**:
   - L2 (Ridge) regularization adds squared penalty: $\lambda \sum \theta_j^2$
   - L1 (Lasso) regularization adds absolute penalty: $\lambda \sum |\theta_j|$
   - Both help prevent overfitting by constraining model complexity

2. **Bias-Variance Tradeoff**:
   - Regularization increases bias but decreases variance
   - Optimal regularization strength minimizes total error
   - Cross-validation helps find this optimal balance

3. **Practical Differences**:
   - **Ridge**: Shrinks coefficients smoothly, good for multicollinearity
   - **Lasso**: Performs feature selection, creates sparse models
   - **Elastic Net**: Combines both, balances shrinkage and selection

4. **Implementation Insights**:
   - Gradient descent works well for regularized optimization
   - Proper feature scaling is crucial
   - Regularization strength (λ) is a critical hyperparameter

5. **When to Use Each**:
   - **Ridge**: When you want to keep all features but reduce their impact
   - **Lasso**: When you want automatic feature selection
   - **Elastic Net**: When you want both shrinkage and selection

### Best Practices

- Always standardize features before applying regularization
- Use cross-validation to select optimal regularization strength
- Consider the interpretability vs. performance tradeoff
- Start with Ridge for stable baseline, try Lasso for feature selection
- Monitor both training and validation performance to detect overfitting

### Further Reading

- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Pattern Recognition and Machine Learning" by Bishop
- scikit-learn documentation on linear models
- Original papers on Lasso regression and Ridge regression