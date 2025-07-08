def backward(self, x):
        """Backward pass using chain rule"""
        # dy/dy = 1
        self.gradients['y'] = 1.0
        
        # dy/dv = dy/dy * d(v^3)/dv = 1 * 3v^2
        self.gradients['v'] = self.gradients['y'] * 3 * (self.values['v']**2)
        
        # dy/du = dy/dv * d(u + 1)/du = dy/dv * 1
        self.gradients['u'] = self.gradients['v'] * 1
        
        # dy/dx = dy/du * d(x^2)/dx = dy/du * 2x
        self.gradients['x'] = self.gradients['u'] * 2 * x
        
        return self.gradients['x']

# Example usage
graph = ComputationGraph()
x = 2.0
y = graph.forward(x)
dy_dx = graph.backward(x)

print(f"f({x}) = {y}")
print(f"f'({x}) = {dy_dx}")

# Verify with analytical derivative: f(x) = (x^2 + 1)^3, f'(x) = 6x(x^2 + 1)^2
analytical_derivative = 6 * x * (x**2 + 1)**2
print(f"Analytical derivative: {analytical_derivative}")
print(f"Numerical verification: {numerical_derivative(lambda x: (x**2 + 1)**3, x)}")
```

### Partial Derivatives and Gradients in ML

```python
# Gradient descent implementation
def gradient_descent_demo():
    """Demonstrate gradient descent optimization"""
    
    # Define a simple loss function: f(x, y) = (x-3)^2 + (y-4)^2
    def loss_function(params):
        x, y = params
        return (x - 3)**2 + (y - 4)**2
    
    def loss_gradient(params):
        """Analytical gradient of the loss function"""
        x, y = params
        return np.array([2*(x - 3), 2*(y - 4)])
    
    # Gradient descent algorithm
    def gradient_descent(initial_params, learning_rate=0.1, max_iterations=100, tolerance=1e-6):
        """Gradient descent optimization"""
        params = initial_params.copy()
        history = [params.copy()]
        
        for i in range(max_iterations):
            grad = loss_gradient(params)
            params = params - learning_rate * grad
            history.append(params.copy())
            
            # Check for convergence
            if np.linalg.norm(grad) < tolerance:
                print(f"Converged after {i+1} iterations")
                break
        
        return params, history
    
    # Run optimization
    initial_params = np.array([0.0, 0.0])
    optimal_params, history = gradient_descent(initial_params)
    
    print(f"Initial parameters: {initial_params}")
    print(f"Optimal parameters: {optimal_params}")
    print(f"True optimum: [3, 4]")
    print(f"Final loss: {loss_function(optimal_params)}")
    
    # Visualize optimization path
    history = np.array(history)
    
    plt.figure(figsize=(10, 8))
    
    # Create contour plot
    x_range = np.linspace(-1, 6, 100)
    y_range = np.linspace(-1, 7, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (X - 3)**2 + (Y - 4)**2
    
    plt.contour(X, Y, Z, levels=20, alpha=0.6)
    plt.colorbar(label='Loss')
    
    # Plot optimization path
    plt.plot(history[:, 0], history[:, 1], 'ro-', linewidth=2, markersize=8, label='Optimization Path')
    plt.plot(3, 4, 'g*', markersize=15, label='True Optimum')
    plt.plot(initial_params[0], initial_params[1], 'bs', markersize=10, label='Start')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return optimal_params, history

gradient_descent_demo()
```

### Higher-Order Derivatives and Hessian Matrix

```python
# Hessian matrix computation
def compute_hessian(f, x, h=1e-5):
    """Compute Hessian matrix numerically"""
    n = len(x)
    hessian = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Compute second partial derivative
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            
            x_pp[i] += h
            x_pp[j] += h
            
            x_pm[i] += h
            x_pm[j] -= h
            
            x_mp[i] -= h
            x_mp[j] += h
            
            x_mm[i] -= h
            x_mm[j] -= h
            
            hessian[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)
    
    return hessian

# Newton's method using Hessian
def newtons_method_demo():
    """Demonstrate Newton's method for optimization"""
    
    # Define function: f(x, y) = x^2 + xy + y^2 - 2x - 3y
    def f(params):
        x, y = params
        return x**2 + x*y + y**2 - 2*x - 3*y
    
    def gradient(params):
        x, y = params
        return np.array([2*x + y - 2, x + 2*y - 3])
    
    def hessian(params):
        # Hessian is constant for this quadratic function
        return np.array([[2, 1], [1, 2]])
    
    def newtons_method(initial_params, max_iterations=10, tolerance=1e-8):
        """Newton's method optimization"""
        params = initial_params.copy()
        history = [params.copy()]
        
        for i in range(max_iterations):
            grad = gradient(params)
            hess = hessian(params)
            
            # Newton's update: x_new = x - H^(-1) * g
            try:
                delta = np.linalg.solve(hess, grad)
                params = params - delta
                history.append(params.copy())
                
                if np.linalg.norm(grad) < tolerance:
                    print(f"Newton's method converged after {i+1} iterations")
                    break
            except np.linalg.LinAlgError:
                print("Hessian is singular, cannot continue")
                break
        
        return params, history
    
    # Compare with gradient descent
    initial_params = np.array([0.0, 0.0])
    
    # Newton's method
    newton_params, newton_history = newtons_method(initial_params)
    
    # Gradient descent for comparison
    def gd_step(params, learning_rate=0.1):
        return params - learning_rate * gradient(params)
    
    gd_params = initial_params.copy()
    gd_history = [gd_params.copy()]
    for _ in range(20):
        gd_params = gd_step(gd_params)
        gd_history.append(gd_params.copy())
    
    print(f"Newton's method result: {newton_params}")
    print(f"Newton's method iterations: {len(newton_history)}")
    print(f"Gradient descent result: {gd_params}")
    print(f"Gradient descent iterations: {len(gd_history)}")
    
    # Analytical solution: solve gradient = 0
    # [2x + y - 2 = 0, x + 2y - 3 = 0] => x = 1/3, y = 4/3
    analytical_solution = np.array([1/3, 4/3])
    print(f"Analytical solution: {analytical_solution}")
    
    return newton_params, newton_history, gd_params, gd_history

newtons_method_demo()
```

---

## 7. Probability and Statistics

### Probability Distributions
Understanding probability distributions is crucial for ML, especially in Bayesian methods and generative models.

```python
# Common probability distributions
import scipy.stats as stats

def probability_distributions_demo():
    """Demonstrate common probability distributions"""
    
    # Normal/Gaussian distribution
    mu, sigma = 0, 1
    normal_dist = stats.norm(mu, sigma)
    
    x = np.linspace(-4, 4, 100)
    normal_pdf = normal_dist.pdf(x)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Normal distribution
    plt.subplot(2, 3, 1)
    plt.plot(x, normal_pdf, 'b-', linewidth=2, label=f'μ={mu}, σ={sigma}')
    plt.title('Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Binomial distribution
    plt.subplot(2, 3, 2)
    n, p = 20, 0.3
    binomial_dist = stats.binom(n, p)
    x_binom = np.arange(0, n+1)
    binomial_pmf = binomial_dist.pmf(x_binom)
    plt.bar(x_binom, binomial_pmf, alpha=0.7, label=f'n={n}, p={p}')
    plt.title('Binomial Distribution')
    plt.xlabel('k')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Poisson distribution
    plt.subplot(2, 3, 3)
    lam = 3
    poisson_dist = stats.poisson(lam)
    x_poisson = np.arange(0, 15)
    poisson_pmf = poisson_dist.pmf(x_poisson)
    plt.bar(x_poisson, poisson_pmf, alpha=0.7, label=f'λ={lam}')
    plt.title('Poisson Distribution')
    plt.xlabel('k')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Exponential distribution
    plt.subplot(2, 3, 4)
    lam_exp = 1.5
    exponential_dist = stats.expon(scale=1/lam_exp)
    x_exp = np.linspace(0, 5, 100)
    exponential_pdf = exponential_dist.pdf(x_exp)
    plt.plot(x_exp, exponential_pdf, 'r-', linewidth=2, label=f'λ={lam_exp}')
    plt.title('Exponential Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Beta distribution
    plt.subplot(2, 3, 5)
    alpha, beta = 2, 5
    beta_dist = stats.beta(alpha, beta)
    x_beta = np.linspace(0, 1, 100)
    beta_pdf = beta_dist.pdf(x_beta)
    plt.plot(x_beta, beta_pdf, 'g-', linewidth=2, label=f'α={alpha}, β={beta}')
    plt.title('Beta Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Gamma distribution
    plt.subplot(2, 3, 6)
    shape, scale = 2, 1
    gamma_dist = stats.gamma(shape, scale=scale)
    x_gamma = np.linspace(0, 8, 100)
    gamma_pdf = gamma_dist.pdf(x_gamma)
    plt.plot(x_gamma, gamma_pdf, 'm-', linewidth=2, label=f'shape={shape}, scale={scale}')
    plt.title('Gamma Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'normal': normal_dist,
        'binomial': binomial_dist,
        'poisson': poisson_dist,
        'exponential': exponential_dist,
        'beta': beta_dist,
        'gamma': gamma_dist
    }

distributions = probability_distributions_demo()
```

### Bayes' Theorem
Bayes' theorem is fundamental to many ML algorithms, especially in classification and Bayesian inference.

```python
# Bayes' theorem implementation and applications
def bayes_theorem_demo():
    """Demonstrate Bayes' theorem with practical examples"""
    
    # Medical diagnosis example
    # P(Disease|Test+) = P(Test+|Disease) * P(Disease) / P(Test+)
    
    # Prior probability of disease
    P_disease = 0.01  # 1% of population has the disease
    
    # Test characteristics
    sensitivity = 0.95  # P(Test+|Disease) = 95%
    specificity = 0.98  # P(Test-|No Disease) = 98%
    
    # Calculate marginal probability P(Test+)
    P_test_positive = sensitivity * P_disease + (1 - specificity) * (1 - P_disease)
    
    # Apply Bayes' theorem
    P_disease_given_positive = (sensitivity * P_disease) / P_test_positive
    
    print("Medical Diagnosis Example:")
    print(f"Prior probability of disease: {P_disease:.3f}")
    print(f"Test sensitivity: {sensitivity:.3f}")
    print(f"Test specificity: {specificity:.3f}")
    print(f"Probability of positive test: {P_test_positive:.3f}")
    print(f"Probability of disease given positive test: {P_disease_given_positive:.3f}")
    
    # Naive Bayes classifier example
    def naive_bayes_classifier():
        """Simple Naive Bayes classifier for text classification"""
        
        # Sample data: word frequencies in spam vs ham emails
        # Features: ['free', 'money', 'click', 'meeting', 'project']
        spam_data = np.array([
            [3, 2, 1, 0, 0],  # Email 1
            [2, 3, 2, 0, 1],  # Email 2
            [1, 1, 3, 0, 0],  # Email 3
            [4, 2, 1, 0, 0],  # Email 4
        ])
        
        ham_data = np.array([
            [0, 0, 0, 2, 3],  # Email 1
            [1, 0, 0, 3, 2],  # Email 2
            [0, 1, 0, 1, 4],  # Email 3
            [0, 0, 1, 2, 3],  # Email 4
        ])
        
        # Calculate class priors
        total_emails = len(spam_data) + len(ham_data)
        P_spam = len(spam_data) / total_emails
        P_ham = len(ham_data) / total_emails
        
        # Calculate likelihoods (using Laplace smoothing)
        alpha = 1  # Smoothing parameter
        vocab_size = spam_data.shape[1]
        
        # Spam likelihoods
        spam_word_counts = np.sum(spam_data, axis=0)
        spam_total_words = np.sum(spam_word_counts)
        spam_likelihoods = (spam_word_counts + alpha) / (spam_total_words + alpha * vocab_size)
        
        # Ham likelihoods
        ham_word_counts = np.sum(ham_data, axis=0)
        ham_total_words = np.sum(ham_word_counts)
        ham_likelihoods = (ham_word_counts + alpha) / (ham_total_words + alpha * vocab_size)
        
        # Classify new email: [1, 1, 0, 1, 2] (word frequencies)
        new_email = np.array([1, 1, 0, 1, 2])
        
        # Calculate posterior probabilities (in log space to avoid underflow)
        log_P_spam_given_email = np.log(P_spam) + np.sum(new_email * np.log(spam_likelihoods))
        log_P_ham_given_email = np.log(P_ham) + np.sum(new_email * np.log(ham_likelihoods))
        
        # Convert back to probabilities
        if log_P_spam_given_email > log_P_ham_given_email:
            prediction = "SPAM"
            confidence = 1 / (1 + np.exp(log_P_ham_given_email - log_P_spam_given_email))
        else:
            prediction = "HAM"
            confidence = 1 / (1 + np.exp(log_P_spam_given_email - log_P_ham_given_email))
        
        print(f"\nNaive Bayes Classifier:")
        print(f"New email word frequencies: {new_email}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.3f}")
        
        return prediction, confidence
    
    naive_bayes_classifier()
    
    return P_disease_given_positive

bayes_result = bayes_theorem_demo()
```

### Statistical Inference
Statistical inference helps us make conclusions about populations from samples.

```python
# Statistical inference examples
def statistical_inference_demo():
    """Demonstrate statistical inference concepts"""
    
    # Generate sample data
    np.random.seed(42)
    population_mean = 50
    population_std = 10
    sample_size = 100
    
    sample = np.random.normal(population_mean, population_std, sample_size)
    
    # Point estimation
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)  # Sample standard deviation
    
    print("Point Estimation:")
    print(f"True population mean: {population_mean}")
    print(f"Sample mean: {sample_mean:.3f}")
    print(f"True population std: {population_std}")
    print(f"Sample std: {sample_std:.3f}")
    
    # Confidence interval for mean
    confidence_level = 0.95
    alpha = 1 - confidence_level
    
    # Using t-distribution for small samples
    t_critical = stats.t.ppf(1 - alpha/2, df=sample_size-1)
    margin_of_error = t_critical * (sample_std / np.sqrt(sample_size))
    
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    print(f"\n{confidence_level*100}% Confidence Interval:")
    print(f"[{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"True mean in CI: {ci_lower <= population_mean <= ci_upper}")
    
    # Hypothesis testing
    # H0: μ = 50, H1: μ ≠ 50
    null_hypothesis_mean = 50
    
    # t-test statistic
    t_stat = (sample_mean - null_hypothesis_mean) / (sample_std / np.sqrt(sample_size))
    
    # p-value (two-tailed test)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=sample_size-1))
    
    print(f"\nHypothesis Test (H0: μ = {null_hypothesis_mean}):")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.3f}")
    print(f"Reject H0 at α = 0.05: {p_value < 0.05}")
    
    # Bootstrap confidence interval
    def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=1000, confidence_level=0.95):
        """Calculate bootstrap confidence interval"""
        n = len(data)
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper, bootstrap_stats
    
    # Bootstrap CI for mean
    boot_ci_lower, boot_ci_upper, boot_stats = bootstrap_confidence_interval(
        sample, np.mean, n_bootstrap=1000
    )
    
    print(f"\nBootstrap {confidence_level*100}% Confidence Interval:")
    print(f"[{boot_ci_lower:.3f}, {boot_ci_upper:.3f}]")
    
    # Visualize bootstrap distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(sample, bins=20, alpha=0.7, density=True, label='Sample Data')
    plt.axvline(sample_mean, color='red', linestyle='--', label=f'Sample Mean: {sample_mean:.2f}')
    plt.axvline(population_mean, color='green', linestyle='--', label=f'True Mean: {population_mean}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Sample Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(boot_stats, bins=30, alpha=0.7, density=True, label='Bootstrap Distribution')
    plt.axvline(boot_ci_lower, color='red', linestyle='--', label=f'CI Lower: {boot_ci_lower:.2f}')
    plt.axvline(boot_ci_upper, color='red', linestyle='--', label=f'CI Upper: {boot_ci_upper:.2f}')
    plt.axvline(population_mean, color='green', linestyle='--', label=f'True Mean: {population_mean}')
    plt.xlabel('Bootstrap Sample Mean')
    plt.ylabel('Density')
    plt.title('Bootstrap Distribution of Sample Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return sample, boot_stats

sample_data, bootstrap_stats = statistical_inference_demo()
```

---

## 8. Optimization Theory

### Gradient Descent Variants
Different gradient descent algorithms for optimizing loss functions.

```python
# Advanced gradient descent algorithms
class OptimizationAlgorithms:
    """Collection of optimization algorithms for ML"""
    
    def __init__(self):
        self.history = []
    
    def rosenbrock_function(self, x):
        """Rosenbrock function: challenging optimization problem"""
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    def rosenbrock_gradient(self, x):
        """Gradient of Rosenbrock function"""
        dx = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        dy = 200 * (x[1] - x[0]**2)
        return np.array([dx, dy])
    
    def gradient_descent(self, initial_x, learning_rate=0.001, max_iter=1000):
        """Standard gradient descent"""
        x = initial_x.copy()
        history = [x.copy()]
        
        for i in range(max_iter):
            grad = self.rosenbrock_gradient(x)
            x = x - learning_rate * grad
            history.append(x.copy())
            
            if np.linalg.norm(grad) < 1e-6:
                break
        
        return x, history
    
    def momentum(self, initial_x, learning_rate=0.001, beta=0.9, max_iter=1000):
        """Gradient descent with momentum"""
        x = initial_x.copy()
        velocity = np.zeros_like(x)
        history = [x.copy()]
        
        for i in range(max_iter):
            grad = self.rosenbrock_gradient(x)
            velocity = beta * velocity + learning_rate * grad
            x = x - velocity
            history.append(x.copy())
            
            if np.linalg.norm(grad) < 1e-6:
                break
        
        return x, history
    
    def adam(self, initial_x, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000):
        """Adam optimizer"""
        x = initial_x.copy()
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment
        history = [x.copy()]
        
        for i in range(1, max_iter + 1):
            grad = self.rosenbrock_gradient(x)
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * grad
            
            # Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * grad**2
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1**i)
            
            # Compute bias-corrected second moment estimate
            v_hat = v / (1 - beta2**i)
            
            # Update parameters
            x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            history.append(x.copy())
            
            if np.linalg.norm(grad) < 1e-6:
                break
        
        return x, history
    
    def rmsprop(self, initial_x, learning_rate=0.001, beta=0.9, epsilon=1e-8, max_iter=1000):
        """RMSprop optimizer"""
        x = initial_x.copy()
        v = np.zeros_like(x)
        history = [x.copy()]
        
        for i in range(max_iter):
            grad = self.rosenbrock_gradient(x)
            
            # Update moving average of squared gradients
            v = beta * v + (1 - beta) * grad**2
            
            # Update parameters
            x = x - learning_rate * grad / (np.sqrt(v) + epsilon)
            history.append(x.copy())
            
            if np.linalg.norm(grad) < 1e-6:
                break
        
        return x, history
    
    def compare_optimizers(self, initial_x=np.array([-1.0, 1.0])):
        """Compare different optimization algorithms"""
        
        # Run different optimizers
        optimizers = {
            'Gradient Descent': lambda: self.gradient_descent(initial_x, learning_rate=0.001),
            'Momentum': lambda: self.momentum(initial_x, learning_rate=0.001),
            'Adam': lambda: self.adam(initial_x, learning_rate=0.01),
            'RMSprop': lambda: self.rmsprop(initial_x, learning_rate=0.01)
        }
        
        results = {}
        for name, optimizer in optimizers.items():
            final_x, history = optimizer()
            results[name] = {
                'final_x': final_x,
                'history': np.array(history),
                'final_loss': self.rosenbrock_function(final_x),
                'iterations': len(history)
            }
        
        # Visualization
        plt.figure(figsize=(15, 10))
        
        # Create contour plot
        x_range = np.linspace(-2, 2, 100)
        y_range = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.rosenbrock_function(np.array([X[i, j], Y[i, j]]))
        
        plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), alpha=0.6)
        plt.colorbar(label='Function Value')
        
        # Plot optimization paths
        colors = ['red', 'blue', 'green', 'orange']
        for i, (name, result) in enumerate(results.items()):
            history = result['history']
            plt.plot(history[:, 0], history[:, 1], colors[i], 
                    linewidth=2, label=f'{name} ({result["iterations"]} iter)')
            plt.plot(history[-1, 0], history[-1, 1], colors[i], 
                    marker='o', markersize=8)
        
        # Mark true optimum
        plt.plot(1, 1, 'k*', markersize=15, label='True Optimum')
        
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.title('Optimization Algorithm Comparison on Rosenbrock Function')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print results
        print("Optimization Results:")
        print("-" * 60)
        for name, result in results.items():
            print(f"{name}:")
            print(f"  Final point: [{result['final