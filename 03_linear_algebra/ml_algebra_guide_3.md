# Mathematical Foundations for ML - Part 2

## Optimization Theory (Continued)

```python
        # Print results
        print("Optimization Results:")
        print("-" * 60)
        for name, result in results.items():
            print(f"{name}:")
            print(f"  Final point: [{result['final_x'][0]:.4f}, {result['final_x'][1]:.4f}]")
            print(f"  Final loss: {result['final_loss']:.6f}")
            print(f"  Iterations: {result['iterations']}")
            print()
        
        return results

# Demonstrate optimization algorithms
optimizer = OptimizationAlgorithms()
optimization_results = optimizer.compare_optimizers()
```

### Constrained Optimization

```python
# Constrained optimization using Lagrange multipliers
def constrained_optimization_demo():
    """Demonstrate constrained optimization"""
    
    # Problem: minimize f(x,y) = x² + y² subject to g(x,y) = x + y - 1 = 0
    def objective(x):
        return x[0]**2 + x[1]**2
    
    def constraint(x):
        return x[0] + x[1] - 1
    
    def constraint_jacobian(x):
        return np.array([1, 1])
    
    # Analytical solution using Lagrange multipliers
    # L(x,y,λ) = x² + y² + λ(x + y - 1)
    # ∇L = [2x + λ, 2y + λ, x + y - 1] = 0
    # Solution: x = y = 1/2, λ = -1
    
    analytical_solution = np.array([0.5, 0.5])
    print("Constrained Optimization Example:")
    print(f"Problem: minimize x² + y² subject to x + y = 1")
    print(f"Analytical solution: x = {analytical_solution[0]}, y = {analytical_solution[1]}")
    print(f"Objective value: {objective(analytical_solution):.4f}")
    
    # Numerical solution using scipy
    from scipy.optimize import minimize
    
    # Define constraint dictionary
    constraint_dict = {
        'type': 'eq',
        'fun': constraint,
        'jac': constraint_jacobian
    }
    
    # Solve numerically
    initial_guess = np.array([0.0, 0.0])
    result = minimize(objective, initial_guess, method='SLSQP', constraints=constraint_dict)
    
    print(f"\nNumerical solution: x = {result.x[0]:.4f}, y = {result.x[1]:.4f}")
    print(f"Objective value: {result.fun:.4f}")
    print(f"Constraint violation: {abs(constraint(result.x)):.6f}")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    # Create contour plot of objective function
    x_range = np.linspace(-1, 2, 100)
    y_range = np.linspace(-1, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + Y**2
    
    plt.contour(X, Y, Z, levels=20, alpha=0.6)
    plt.colorbar(label='Objective Value')
    
    # Plot constraint line
    x_constraint = np.linspace(-1, 2, 100)
    y_constraint = 1 - x_constraint
    plt.plot(x_constraint, y_constraint, 'r-', linewidth=3, label='Constraint: x + y = 1')
    
    # Mark solutions
    plt.plot(analytical_solution[0], analytical_solution[1], 'go', 
             markersize=10, label='Analytical Solution')
    plt.plot(result.x[0], result.x[1], 'bo', 
             markersize=10, label='Numerical Solution')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Constrained Optimization: Minimize x² + y² subject to x + y = 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    
    return result

constrained_result = constrained_optimization_demo()
```

### Convex Optimization

```python
# Convex optimization concepts
def convex_optimization_demo():
    """Demonstrate convex optimization concepts"""
    
    # Example: Portfolio optimization (quadratic programming)
    # Minimize: (1/2) * x^T * Q * x - μ^T * x
    # Subject to: sum(x) = 1, x ≥ 0
    
    np.random.seed(42)
    n_assets = 4
    
    # Generate random expected returns
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)
    
    # Generate random covariance matrix (positive semi-definite)
    A = np.random.randn(n_assets, n_assets)
    covariance_matrix = A @ A.T
    
    print("Portfolio Optimization Example:")
    print(f"Number of assets: {n_assets}")
    print(f"Expected returns: {expected_returns}")
    print(f"Covariance matrix shape: {covariance_matrix.shape}")
    
    # Solve for different risk aversion parameters
    risk_aversions = [0.5, 1.0, 2.0, 5.0]
    efficient_frontier = []
    
    for risk_aversion in risk_aversions:
        # Objective: minimize (1/2) * risk_aversion * x^T * Σ * x - μ^T * x
        def portfolio_objective(x):
            return 0.5 * risk_aversion * x.T @ covariance_matrix @ x - expected_returns.T @ x
        
        def portfolio_gradient(x):
            return risk_aversion * covariance_matrix @ x - expected_returns
        
        # Constraints: sum(x) = 1, x ≥ 0
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Solve optimization problem
        initial_guess = np.ones(n_assets) / n_assets
        result = minimize(portfolio_objective, initial_guess, method='SLSQP',
                         jac=portfolio_gradient, constraints=constraints, bounds=bounds)
        
        if result.success:
            portfolio_return = expected_returns.T @ result.x
            portfolio_risk = np.sqrt(result.x.T @ covariance_matrix @ result.x)
            efficient_frontier.append((portfolio_risk, portfolio_return, result.x))
    
    # Plot efficient frontier
    plt.figure(figsize=(12, 8))
    
    risks = [point[0] for point in efficient_frontier]
    returns = [point[1] for point in efficient_frontier]
    
    plt.subplot(1, 2, 1)
    plt.plot(risks, returns, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.grid(True, alpha=0.3)
    
    # Plot portfolio weights
    plt.subplot(1, 2, 2)
    weights_matrix = np.array([point[2] for point in efficient_frontier])
    
    bottom = np.zeros(len(risk_aversions))
    colors = ['red', 'blue', 'green', 'orange']
    
    for i in range(n_assets):
        plt.bar(range(len(risk_aversions)), weights_matrix[:, i], 
                bottom=bottom, label=f'Asset {i+1}', color=colors[i], alpha=0.7)
        bottom += weights_matrix[:, i]
    
    plt.xlabel('Risk Aversion Parameter')
    plt.ylabel('Portfolio Weight')
    plt.title('Portfolio Composition')
    plt.xticks(range(len(risk_aversions)), [f'{ra}' for ra in risk_aversions])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return efficient_frontier

efficient_frontier = convex_optimization_demo()
```

---

## 9. Information Theory

### Entropy and Information

```python
# Information theory concepts
def information_theory_demo():
    """Demonstrate information theory concepts"""
    
    # Shannon entropy
    def shannon_entropy(probabilities):
        """Calculate Shannon entropy"""
        # Remove zero probabilities to avoid log(0)
        p = np.array(probabilities)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    # Example: Coin flip entropy
    print("Information Theory Concepts:")
    print("="*50)
    
    # Fair coin
    fair_coin = [0.5, 0.5]
    fair_entropy = shannon_entropy(fair_coin)
    print(f"Fair coin entropy: {fair_entropy:.3f} bits")
    
    # Biased coin
    biased_coin = [0.9, 0.1]
    biased_entropy = shannon_entropy(biased_coin)
    print(f"Biased coin entropy: {biased_entropy:.3f} bits")
    
    # Uniform distribution over 8 outcomes
    uniform_8 = [1/8] * 8
    uniform_entropy = shannon_entropy(uniform_8)
    print(f"Uniform distribution (8 outcomes) entropy: {uniform_entropy:.3f} bits")
    
    # Cross-entropy
    def cross_entropy(true_probs, predicted_probs):
        """Calculate cross-entropy"""
        true_probs = np.array(true_probs)
        predicted_probs = np.array(predicted_probs)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
        return -np.sum(true_probs * np.log2(predicted_probs))
    
    # KL divergence
    def kl_divergence(true_probs, predicted_probs):
        """Calculate KL divergence"""
        return cross_entropy(true_probs, predicted_probs) - shannon_entropy(true_probs)
    
    # Example: Classification with 3 classes
    true_distribution = [0.7, 0.2, 0.1]
    predicted_distribution = [0.6, 0.3, 0.1]
    
    ce = cross_entropy(true_distribution, predicted_distribution)
    kl = kl_divergence(true_distribution, predicted_distribution)
    
    print(f"\nClassification Example:")
    print(f"True distribution: {true_distribution}")
    print(f"Predicted distribution: {predicted_distribution}")
    print(f"Cross-entropy: {ce:.3f} bits")
    print(f"KL divergence: {kl:.3f} bits")
    
    # Mutual information
    def mutual_information_discrete(X, Y):
        """Calculate mutual information for discrete variables"""
        # Create joint distribution
        unique_x = np.unique(X)
        unique_y = np.unique(Y)
        
        joint_probs = np.zeros((len(unique_x), len(unique_y)))
        marginal_x = np.zeros(len(unique_x))
        marginal_y = np.zeros(len(unique_y))
        
        n_samples = len(X)
        
        for i, x_val in enumerate(unique_x):
            for j, y_val in enumerate(unique_y):
                joint_probs[i, j] = np.sum((X == x_val) & (Y == y_val)) / n_samples
                
        marginal_x = np.sum(joint_probs, axis=1)
        marginal_y = np.sum(joint_probs, axis=0)
        
        # Calculate mutual information
        mi = 0
        for i in range(len(unique_x)):
            for j in range(len(unique_y)):
                if joint_probs[i, j] > 0:
                    mi += joint_probs[i, j] * np.log2(joint_probs[i, j] / (marginal_x[i] * marginal_y[j]))
        
        return mi
    
    # Generate correlated discrete data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randint(0, 3, n_samples)
    Y = (X + np.random.randint(0, 2, n_samples)) % 3  # Y is correlated with X
    
    mi = mutual_information_discrete(X, Y)
    print(f"\nMutual Information Example:")
    print(f"Mutual information I(X;Y): {mi:.3f} bits")
    
    # Visualization of entropy vs probability
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Entropy vs probability for binary distribution
    plt.subplot(2, 3, 1)
    p_values = np.linspace(0.01, 0.99, 100)
    entropies = [-p * np.log2(p) - (1-p) * np.log2(1-p) for p in p_values]
    plt.plot(p_values, entropies, 'b-', linewidth=2)
    plt.xlabel('Probability p')
    plt.ylabel('Entropy (bits)')
    plt.title('Binary Entropy Function')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cross-entropy vs predicted probability
    plt.subplot(2, 3, 2)
    true_p = 0.7
    pred_p_values = np.linspace(0.01, 0.99, 100)
    ce_values = [-true_p * np.log2(p) - (1-true_p) * np.log2(1-p) for p in pred_p_values]
    plt.plot(pred_p_values, ce_values, 'r-', linewidth=2)
    plt.axvline(true_p, color='g', linestyle='--', label=f'True p = {true_p}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Cross-entropy (bits)')
    plt.title('Cross-entropy vs Predicted Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: KL divergence
    plt.subplot(2, 3, 3)
    kl_values = [ce - shannon_entropy([true_p, 1-true_p]) for ce in ce_values]
    plt.plot(pred_p_values, kl_values, 'purple', linewidth=2)
    plt.axvline(true_p, color='g', linestyle='--', label=f'True p = {true_p}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('KL Divergence (bits)')
    plt.title('KL Divergence vs Predicted Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Joint distribution heatmap
    plt.subplot(2, 3, 4)
    joint_hist = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            joint_hist[i, j] = np.sum((X == i) & (Y == j))
    
    plt.imshow(joint_hist, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Count')
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title('Joint Distribution of X and Y')
    plt.xticks(range(3))
    plt.yticks(range(3))
    
    # Plot 5: Marginal distributions
    plt.subplot(2, 3, 5)
    x_counts = np.bincount(X)
    y_counts = np.bincount(Y)
    
    x_positions = np.arange(3)
    width = 0.35
    
    plt.bar(x_positions - width/2, x_counts, width, label='X', alpha=0.7)
    plt.bar(x_positions + width/2, y_counts, width, label='Y', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Marginal Distributions')
    plt.legend()
    plt.xticks(x_positions)
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Information theory measures comparison
    plt.subplot(2, 3, 6)
    h_x = shannon_entropy(x_counts / np.sum(x_counts))
    h_y = shannon_entropy(y_counts / np.sum(y_counts))
    
    measures = ['H(X)', 'H(Y)', 'I(X;Y)']
    values = [h_x, h_y, mi]
    
    plt.bar(measures, values, color=['blue', 'red', 'green'], alpha=0.7)
    plt.ylabel('Information (bits)')
    plt.title('Information Theory Measures')
    plt.grid(True, alpha=0.3)
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'entropy_measures': {
            'fair_coin': fair_entropy,
            'biased_coin': biased_entropy,
            'uniform_8': uniform_entropy
        },
        'cross_entropy': ce,
        'kl_divergence': kl,
        'mutual_information': mi
    }

info_theory_results = information_theory_demo()
```

### Decision Trees and Information Gain

```python
# Information gain in decision trees
def decision_tree_info_gain_demo():
    """Demonstrate information gain in decision tree construction"""
    
    # Create sample dataset
    # Features: [Outlook, Temperature, Humidity, Wind]
    # Target: Play Tennis (Yes/No)
    
    data = np.array([
        ['Sunny', 'Hot', 'High', 'Weak', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Sunny', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Strong', 'No']
    ])
    
    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    target = data[:, -1]
    
    def calculate_entropy(labels):
        """Calculate entropy of label distribution"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def calculate_information_gain(data, feature_idx, target_idx):
        """Calculate information gain for a feature"""
        target_values = data[:, target_idx]
        feature_values = data[:, feature_idx]
        
        # Calculate entropy of target
        total_entropy = calculate_entropy(target_values)
        
        # Calculate weighted entropy after split
        unique_features = np.unique(feature_values)
        weighted_entropy = 0
        
        for feature_val in unique_features:
            subset_mask = feature_values == feature_val
            subset_target = target_values[subset_mask]
            subset_weight = len(subset_target) / len(target_values)
            subset_entropy = calculate_entropy(subset_target)
            weighted_entropy += subset_weight * subset_entropy
        
        # Information gain = Total entropy - Weighted entropy
        information_gain = total_entropy - weighted_entropy
        
        return information_gain, total_entropy, weighted_entropy
    
    # Calculate information gain for each feature
    print("Decision Tree Information Gain Analysis:")
    print("="*50)
    
    target_entropy = calculate_entropy(target)
    print(f"Target entropy: {target_entropy:.3f}")
    print()
    
    gains = []
    for i, feature in enumerate(features):
        ig, total_ent, weighted_ent = calculate_information_gain(data, i, -1)
        gains.append((feature, ig, total_ent, weighted_ent))
        print(f"Feature: {feature}")
        print(f"  Information Gain: {ig:.3f}")
        print(f"  Weighted Entropy: {weighted_ent:.3f}")
        print()
    
    # Sort by information gain
    gains.sort(key=lambda x: x[1], reverse=True)
    best_feature = gains[0][0]
    print(f"Best feature to split on: {best_feature}")
    
    # Visualize information gain
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Information gain comparison
    plt.subplot(2, 2, 1)
    feature_names = [g[0] for g in gains]
    ig_values = [g[1] for g in gains]
    
    bars = plt.bar(feature_names, ig_values, color=['red' if f == best_feature else 'blue' for f in feature_names])
    plt.ylabel('Information Gain')
    plt.title('Information Gain by Feature')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Highlight best feature
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 2: Target distribution
    plt.subplot(2, 2, 2)
    target_counts = np.unique(target, return_counts=True)
    plt.pie(target_counts[1], labels=target_counts[0], autopct='%1.1f%%', startangle=90)
    plt.title('Target Distribution')
    
    # Plot 3: Best feature distribution
    plt.subplot(2, 2, 3)
    best_feature_idx = features.index(best_feature)
    best_feature_values = data[:, best_feature_idx]
    
    # Create contingency table
    unique_features = np.unique(best_feature_values)
    unique_targets = np.unique(target)
    
    contingency = np.zeros((len(unique_features), len(unique_targets)))
    for i, feat_val in enumerate(unique_features):
        for j, target_val in enumerate(unique_targets):
            contingency[i, j] = np.sum((best_feature_values == feat_val) & (target == target_val))
    
    # Stacked bar chart
    bottom = np.zeros(len(unique_features))
    colors = ['red', 'blue']
    
    for j, target_val in enumerate(unique_targets):
        plt.bar(unique_features, contingency[:, j], bottom=bottom, 
                label=f'Target = {target_val}', color=colors[j], alpha=0.7)
        bottom += contingency[:, j]
    
    plt.xlabel(best_feature)
    plt.ylabel('Count')
    plt.title(f'Distribution of {best_feature} vs Target')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Entropy reduction visualization
    plt.subplot(2, 2, 4)
    
    # Calculate entropy for each split
    entropies_before = [target_entropy] * len(unique_features)
    entropies_after = []
    
    for feat_val in unique_features:
        subset_mask = best_feature_values == feat_val
        subset_target = target[subset_mask]
        subset_entropy = calculate_entropy(subset_target)
        entropies_after.append(subset_entropy)
    
    x_pos = np.arange(len(unique_features))
    width = 0.35
    
    plt.bar(x_pos - width/2, entropies_before, width, label='Before Split', alpha=0.7)
    plt.bar(x_pos + width/2, entropies_after, width, label='After Split', alpha=0.7)
    
    plt.xlabel(f'{best_feature} Values')
    plt.ylabel('Entropy')
    plt.title('Entropy Before and After Split')
    plt.xticks(x_pos, unique_features, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return gains, best_feature

dt_results = decision_tree_info_gain_demo()
```

---

## 10. Dimensionality and Complexity

### Curse of Dimensionality

```python
# Demonstrate curse of dimensionality
def curse_of_dimensionality_demo():
    """Demonstrate the curse of dimensionality"""
    
    # Distance concentration in high dimensions
    def distance_concentration():
        """Show how distances concentrate in high dimensions"""
        dimensions = [2, 5, 10, 20, 50, 100]
        n_samples = 1000
        results = []
        
        np.random.seed(42)
        
        for d in dimensions:
            # Generate random points in d-dimensional unit hypercube
            points = np.random.uniform(0, 1, (n_samples, d))
            
            # Calculate pairwise distances
            distances = []
            for i in range(100):  # Sample subset for efficiency
                for j in range(i+1, 100):
                    dist = np.linalg.norm(points[i] - points[j])
                    distances.append(dist)
            
            distances = np.array(distances)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            coeff_var = std_dist / mean_dist
            
            results.append({
                'dimension': d,
                'mean_distance': mean_dist,
                'std_distance': std_dist,
                'coefficient_of_variation': coeff_var
            })
        
        return results
    
    # Volume of unit hypersphere
    def hypersphere_volume():
        """Calculate volume of unit hypersphere in different dimensions"""
        from math import gamma, pi
        
        dimensions = np.arange(1, 21)
        volumes = []
        
        for d in dimensions:
            # Volume of d-dimensional unit hypersphere
            volume = (pi**(d/2)) / gamma(d/2 + 1)
            volumes.append(volume)
        
        return dimensions, volumes
    
    # Sample density in high dimensions
    def sample_density():
        """Show how sample density decreases with dimension"""
        dimensions = [2, 5, 10, 20]
        n_samples = 1000
        results = []
        
        for d in dimensions:
            # Volume of unit hypercube
            volume = 1.0  # Unit hypercube has volume 1
            
            # Expected distance between samples
            expected_dist = (volume / n_samples) ** (1/d)
            
            results.append({
                'dimension': d,
                'volume': volume,
                'expected_distance': expected_dist
            })
        
        return results
    
    # Run analyses
    print("Curse of Dimensionality Analysis:")
    print("="*50)
    
    # Distance concentration
    dist_results = distance_concentration()
    print("\nDistance Concentration:")
    for result in dist_results:
        print(f"Dimension {result['dimension']:3d}: "
              f"Mean = {result['mean_distance']:.3f}, "
              f"Std = {result['std_distance']:.3f}, "
              f"CV = {result['coefficient_of_variation']:.3f}")
    
    # Hypersphere volume
    dims, volumes = hypersphere_volume()
    print(f"\nHypersphere Volume:")
    for d, v in zip(dims[:10], volumes[:10]):
        print(f"Dimension {d:2d}: Volume = {v:.6f}")
    
    # Sample density
    density_results = sample_density()
    print(f"\nSample Density (1000 samples):")
    for result in density_results:
        print(f"Dimension {result['dimension']:2d}: "
              f"Expected distance = {result['expected_distance']:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distance concentration
    plt.subplot(2, 3, 1)
    dims = [r['dimension'] for r in dist_results]
    cvs = [r['coefficient_of_variation'] for r in dist_results]
    
    plt.plot(dims, cvs, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Dimension')
    plt.ylabel('Coefficient of Variation')
    plt.title('Distance Concentration')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Hypersphere volume
    plt.subplot(2, 3, 2)
    plt.plot(dims, volumes, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Dimension')
    plt.ylabel('Volume')
    plt.title('Unit Hypersphere Volume')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Sample density
    plt.subplot(2, 3, 3)
    density_dims = [r['dimension'] for r in density_results]
    expected_dists = [r['expected_distance'] for r in density_results]
    
    plt.plot(density_dims, expected_dists, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Dimension')
    plt.ylabel('Expected Distance Between Samples')