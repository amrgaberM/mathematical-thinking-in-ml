# Probability & Statistics for Machine Learning - Complete Guide

## Table of Contents
1. [Introduction & Setup](#introduction)
2. [Basic Probability Concepts](#basic-probability)
3. [Probability Distributions](#distributions)
4. [Descriptive Statistics](#descriptive-stats)
5. [Inferential Statistics](#inferential-stats)
6. [Bayesian Statistics](#bayesian)
7. [Hypothesis Testing](#hypothesis-testing)
8. [Correlation & Regression](#correlation-regression)
9. [ML-Specific Applications](#ml-applications)
10. [Practice Problems](#practice)

---

## 1. Introduction & Setup {#introduction}

### Why Probability & Statistics Matter in ML

Probability and statistics form the mathematical foundation of machine learning. They help us:
- Understand uncertainty in data and predictions
- Make informed decisions with incomplete information
- Evaluate model performance and reliability
- Design algorithms that learn from data
- Handle noise and variability in real-world datasets

```python
# Essential imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform, expon
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

---

## 2. Basic Probability Concepts {#basic-probability}

### 2.1 Sample Space and Events

**Sample Space (Ω)**: Set of all possible outcomes
**Event (A)**: Subset of sample space

```python
# Example: Coin flip
sample_space_coin = ['H', 'T']
event_heads = ['H']

# Example: Die roll
sample_space_die = [1, 2, 3, 4, 5, 6]
event_even = [2, 4, 6]

print(f"Coin sample space: {sample_space_coin}")
print(f"Die sample space: {sample_space_die}")
print(f"Event 'even number': {event_even}")
```

### 2.2 Probability Axioms

1. **Non-negativity**: P(A) ≥ 0 for any event A
2. **Normalization**: P(Ω) = 1
3. **Additivity**: P(A ∪ B) = P(A) + P(B) for disjoint events

```python
def probability_axioms_demo():
    # Simulate 10000 coin flips
    flips = np.random.choice(['H', 'T'], size=10000)
    
    # Calculate probabilities
    p_heads = np.mean(flips == 'H')
    p_tails = np.mean(flips == 'T')
    
    print(f"P(Heads) = {p_heads:.3f}")
    print(f"P(Tails) = {p_tails:.3f}")
    print(f"P(Heads) + P(Tails) = {p_heads + p_tails:.3f}")
    
    # Verify axioms
    print(f"\nAxiom 1 (Non-negativity): P(H) ≥ 0? {p_heads >= 0}")
    print(f"Axiom 2 (Normalization): P(H) + P(T) = 1? {abs(p_heads + p_tails - 1) < 0.01}")

probability_axioms_demo()
```

### 2.3 Conditional Probability

**Definition**: P(A|B) = P(A ∩ B) / P(B)

```python
def conditional_probability_demo():
    # Medical test example
    # P(Disease) = 0.01 (1% of population has disease)
    # P(Positive|Disease) = 0.95 (95% sensitivity)
    # P(Positive|No Disease) = 0.05 (5% false positive rate)
    
    p_disease = 0.01
    p_pos_given_disease = 0.95
    p_pos_given_no_disease = 0.05
    
    # Calculate P(Positive)
    p_positive = p_pos_given_disease * p_disease + p_pos_given_no_disease * (1 - p_disease)
    
    # Calculate P(Disease|Positive) using Bayes' theorem
    p_disease_given_positive = (p_pos_given_disease * p_disease) / p_positive
    
    print(f"P(Disease) = {p_disease:.3f}")
    print(f"P(Positive|Disease) = {p_pos_given_disease:.3f}")
    print(f"P(Positive|No Disease) = {p_pos_given_no_disease:.3f}")
    print(f"P(Positive) = {p_positive:.3f}")
    print(f"P(Disease|Positive) = {p_disease_given_positive:.3f}")

conditional_probability_demo()
```

### 2.4 Independence

Events A and B are independent if P(A|B) = P(A), or equivalently, P(A ∩ B) = P(A)P(B)

```python
def independence_demo():
    # Simulate two independent coin flips
    n_trials = 10000
    coin1 = np.random.choice([0, 1], size=n_trials)  # 0=T, 1=H
    coin2 = np.random.choice([0, 1], size=n_trials)
    
    # Calculate probabilities
    p_h1 = np.mean(coin1 == 1)
    p_h2 = np.mean(coin2 == 1)
    p_both_h = np.mean((coin1 == 1) & (coin2 == 1))
    
    print(f"P(Coin1 = H) = {p_h1:.3f}")
    print(f"P(Coin2 = H) = {p_h2:.3f}")
    print(f"P(Both H) = {p_both_h:.3f}")
    print(f"P(Coin1 = H) × P(Coin2 = H) = {p_h1 * p_h2:.3f}")
    print(f"Independent? {abs(p_both_h - p_h1 * p_h2) < 0.01}")

independence_demo()
```

---

## 3. Probability Distributions {#distributions}

### 3.1 Discrete Distributions

#### Bernoulli Distribution
Single trial with two outcomes (success/failure)

```python
def bernoulli_demo():
    p = 0.3  # probability of success
    
    # Generate samples
    samples = np.random.binomial(1, p, size=1000)
    
    # Calculate statistics
    mean_theoretical = p
    variance_theoretical = p * (1 - p)
    mean_empirical = np.mean(samples)
    variance_empirical = np.var(samples)
    
    print(f"Bernoulli Distribution (p = {p})")
    print(f"Theoretical: Mean = {mean_theoretical:.3f}, Variance = {variance_theoretical:.3f}")
    print(f"Empirical: Mean = {mean_empirical:.3f}, Variance = {variance_empirical:.3f}")
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.hist(samples, bins=[-0.5, 0.5, 1.5], alpha=0.7, density=True)
    plt.xlabel('Outcome')
    plt.ylabel('Probability')
    plt.title('Bernoulli Distribution')
    plt.xticks([0, 1])
    plt.show()

bernoulli_demo()
```

#### Binomial Distribution
Number of successes in n independent Bernoulli trials

```python
def binomial_demo():
    n, p = 20, 0.3
    
    # Generate samples
    samples = np.random.binomial(n, p, size=1000)
    
    # Calculate statistics
    mean_theoretical = n * p
    variance_theoretical = n * p * (1 - p)
    mean_empirical = np.mean(samples)
    variance_empirical = np.var(samples)
    
    print(f"Binomial Distribution (n = {n}, p = {p})")
    print(f"Theoretical: Mean = {mean_theoretical:.3f}, Variance = {variance_theoretical:.3f}")
    print(f"Empirical: Mean = {mean_empirical:.3f}, Variance = {variance_empirical:.3f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.hist(samples, bins=range(n+2), alpha=0.7, density=True, label='Empirical')
    
    # Theoretical PMF
    x = np.arange(0, n+1)
    theoretical_pmf = binom.pmf(x, n, p)
    plt.plot(x, theoretical_pmf, 'ro-', label='Theoretical')
    
    plt.xlabel('Number of Successes')
    plt.ylabel('Probability')
    plt.title('Binomial Distribution')
    plt.legend()
    plt.show()

binomial_demo()
```

#### Poisson Distribution
Number of events in a fixed interval (rare events)

```python
def poisson_demo():
    lam = 3  # average rate
    
    # Generate samples
    samples = np.random.poisson(lam, size=1000)
    
    # Calculate statistics
    mean_theoretical = lam
    variance_theoretical = lam
    mean_empirical = np.mean(samples)
    variance_empirical = np.var(samples)
    
    print(f"Poisson Distribution (λ = {lam})")
    print(f"Theoretical: Mean = {mean_theoretical:.3f}, Variance = {variance_theoretical:.3f}")
    print(f"Empirical: Mean = {mean_empirical:.3f}, Variance = {variance_empirical:.3f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.hist(samples, bins=range(0, 15), alpha=0.7, density=True, label='Empirical')
    
    # Theoretical PMF
    x = np.arange(0, 15)
    theoretical_pmf = poisson.pmf(x, lam)
    plt.plot(x, theoretical_pmf, 'ro-', label='Theoretical')
    
    plt.xlabel('Number of Events')
    plt.ylabel('Probability')
    plt.title('Poisson Distribution')
    plt.legend()
    plt.show()

poisson_demo()
```

### 3.2 Continuous Distributions

#### Uniform Distribution
All values in an interval are equally likely

```python
def uniform_demo():
    a, b = 0, 10  # interval [a, b]
    
    # Generate samples
    samples = np.random.uniform(a, b, size=1000)
    
    # Calculate statistics
    mean_theoretical = (a + b) / 2
    variance_theoretical = (b - a)**2 / 12
    mean_empirical = np.mean(samples)
    variance_empirical = np.var(samples)
    
    print(f"Uniform Distribution [a = {a}, b = {b}]")
    print(f"Theoretical: Mean = {mean_theoretical:.3f}, Variance = {variance_theoretical:.3f}")
    print(f"Empirical: Mean = {mean_empirical:.3f}, Variance = {variance_empirical:.3f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.hist(samples, bins=30, alpha=0.7, density=True, label='Empirical')
    
    # Theoretical PDF
    x = np.linspace(a-1, b+1, 1000)
    theoretical_pdf = uniform.pdf(x, a, b-a)
    plt.plot(x, theoretical_pdf, 'r-', linewidth=2, label='Theoretical')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Uniform Distribution')
    plt.legend()
    plt.show()

uniform_demo()
```

#### Normal (Gaussian) Distribution
Bell-shaped curve, fundamental in statistics

```python
def normal_demo():
    mu, sigma = 0, 1  # mean and standard deviation
    
    # Generate samples
    samples = np.random.normal(mu, sigma, size=1000)
    
    # Calculate statistics
    mean_theoretical = mu
    variance_theoretical = sigma**2
    mean_empirical = np.mean(samples)
    variance_empirical = np.var(samples)
    
    print(f"Normal Distribution (μ = {mu}, σ = {sigma})")
    print(f"Theoretical: Mean = {mean_theoretical:.3f}, Variance = {variance_theoretical:.3f}")
    print(f"Empirical: Mean = {mean_empirical:.3f}, Variance = {variance_empirical:.3f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.hist(samples, bins=30, alpha=0.7, density=True, label='Empirical')
    
    # Theoretical PDF
    x = np.linspace(-4, 4, 1000)
    theoretical_pdf = norm.pdf(x, mu, sigma)
    plt.plot(x, theoretical_pdf, 'r-', linewidth=2, label='Theoretical')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Normal Distribution')
    plt.legend()
    plt.show()

normal_demo()
```

#### Exponential Distribution
Time between events in a Poisson process

```python
def exponential_demo():
    lam = 1  # rate parameter
    
    # Generate samples
    samples = np.random.exponential(1/lam, size=1000)
    
    # Calculate statistics
    mean_theoretical = 1/lam
    variance_theoretical = 1/lam**2
    mean_empirical = np.mean(samples)
    variance_empirical = np.var(samples)
    
    print(f"Exponential Distribution (λ = {lam})")
    print(f"Theoretical: Mean = {mean_theoretical:.3f}, Variance = {variance_theoretical:.3f}")
    print(f"Empirical: Mean = {mean_empirical:.3f}, Variance = {variance_empirical:.3f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.hist(samples, bins=30, alpha=0.7, density=True, label='Empirical')
    
    # Theoretical PDF
    x = np.linspace(0, 6, 1000)
    theoretical_pdf = expon.pdf(x, scale=1/lam)
    plt.plot(x, theoretical_pdf, 'r-', linewidth=2, label='Theoretical')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Exponential Distribution')
    plt.legend()
    plt.show()

exponential_demo()
```

### 3.3 Central Limit Theorem

```python
def central_limit_theorem_demo():
    # Sample from non-normal distribution (exponential)
    population_dist = lambda size: np.random.exponential(2, size)
    
    sample_sizes = [1, 5, 10, 30, 100]
    n_samples = 1000
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, n in enumerate(sample_sizes):
        # Generate sample means
        sample_means = []
        for _ in range(n_samples):
            sample = population_dist(n)
            sample_means.append(np.mean(sample))
        
        # Plot histogram
        axes[i].hist(sample_means, bins=30, alpha=0.7, density=True)
        axes[i].set_title(f'Sample Size n = {n}')
        axes[i].set_xlabel('Sample Mean')
        axes[i].set_ylabel('Density')
        
        # Overlay theoretical normal distribution
        theoretical_mean = 2  # mean of exponential(2)
        theoretical_std = 2 / np.sqrt(n)  # std of sample mean
        x = np.linspace(np.min(sample_means), np.max(sample_means), 100)
        theoretical_pdf = norm.pdf(x, theoretical_mean, theoretical_std)
        axes[i].plot(x, theoretical_pdf, 'r-', linewidth=2, label='Theoretical Normal')
        axes[i].legend()
    
    # Show original population distribution
    original_sample = population_dist(10000)
    axes[5].hist(original_sample, bins=50, alpha=0.7, density=True)
    axes[5].set_title('Original Population (Exponential)')
    axes[5].set_xlabel('Value')
    axes[5].set_ylabel('Density')
    
    plt.tight_layout()
    plt.show()

central_limit_theorem_demo()
```

---

## 4. Descriptive Statistics {#descriptive-stats}

### 4.1 Measures of Central Tendency

```python
def central_tendency_demo():
    # Generate sample data
    np.random.seed(42)
    data = np.random.normal(50, 15, 1000)
    
    # Add some outliers
    data = np.concatenate([data, [100, 105, 110]])
    
    # Calculate measures
    mean = np.mean(data)
    median = np.median(data)
    mode_value = stats.mode(data.round()).mode[0]
    
    print(f"Dataset size: {len(data)}")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode_value:.2f}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7, density=True)
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean:.2f}')
    plt.axvline(median, color='green', linestyle='--', label=f'Median = {median:.2f}')
    plt.axvline(mode_value, color='blue', linestyle='--', label=f'Mode = {mode_value:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution with Outliers')
    plt.legend()
    plt.show()

central_tendency_demo()
```

### 4.2 Measures of Variability

```python
def variability_demo():
    # Generate sample data
    np.random.seed(42)
    data = np.random.normal(50, 15, 1000)
    
    # Calculate measures
    variance = np.var(data, ddof=1)  # sample variance
    std_dev = np.std(data, ddof=1)  # sample standard deviation
    range_val = np.max(data) - np.min(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    mad = np.median(np.abs(data - np.median(data)))  # Median Absolute Deviation
    
    print(f"Variance: {variance:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Range: {range_val:.2f}")
    print(f"Interquartile Range (IQR): {iqr:.2f}")
    print(f"Median Absolute Deviation (MAD): {mad:.2f}")
    
    # Box plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.boxplot(data)
    plt.title('Box Plot')
    plt.ylabel('Value')
    
    # Histogram with std dev markers
    plt.subplot(1, 2, 2)
    plt.hist(data, bins=30, alpha=0.7, density=True)
    mean = np.mean(data)
    plt.axvline(mean, color='red', linestyle='-', label=f'Mean = {mean:.2f}')
    plt.axvline(mean - std_dev, color='orange', linestyle='--', label=f'Mean - 1σ')
    plt.axvline(mean + std_dev, color='orange', linestyle='--', label=f'Mean + 1σ')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution with Standard Deviation')
    plt.legend()
    plt.tight_layout()
    plt.show()

variability_demo()
```

### 4.3 Shape of Distribution

```python
def distribution_shape_demo():
    # Generate different distributions
    np.random.seed(42)
    
    # Normal distribution
    normal_data = np.random.normal(0, 1, 1000)
    
    # Right-skewed distribution
    right_skewed = np.random.exponential(2, 1000)
    
    # Left-skewed distribution
    left_skewed = -np.random.exponential(2, 1000)
    
    # Calculate skewness and kurtosis
    datasets = [normal_data, right_skewed, left_skewed]
    names = ['Normal', 'Right-skewed', 'Left-skewed']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (data, name) in enumerate(zip(datasets, names)):
        skewness = stats.skew(data)
        kurt = stats.kurtosis(data)
        
        axes[i].hist(data, bins=30, alpha=0.7, density=True)
        axes[i].set_title(f'{name}\nSkewness: {skewness:.2f}, Kurtosis: {kurt:.2f}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        
        print(f"{name} Distribution:")
        print(f"  Skewness: {skewness:.3f}")
        print(f"  Kurtosis: {kurt:.3f}")
        print()
    
    plt.tight_layout()
    plt.show()

distribution_shape_demo()
```

---

## 5. Inferential Statistics {#inferential-stats}

### 5.1 Confidence Intervals

```python
def confidence_interval_demo():
    # Generate sample data
    np.random.seed(42)
    population_mean = 50
    population_std = 10
    sample_size = 100
    
    sample = np.random.normal(population_mean, population_std, sample_size)
    
    # Calculate confidence intervals
    confidence_levels = [0.90, 0.95, 0.99]
    
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    standard_error = sample_std / np.sqrt(sample_size)
    
    print(f"Sample mean: {sample_mean:.2f}")
    print(f"Sample standard deviation: {sample_std:.2f}")
    print(f"Standard error: {standard_error:.2f}")
    print(f"True population mean: {population_mean}")
    print()
    
    for confidence_level in confidence_levels:
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df=sample_size-1)
        
        margin_of_error = t_critical * standard_error
        ci_lower = sample_mean - margin_of_error
        ci_upper = sample_mean + margin_of_error
        
        contains_true = ci_lower <= population_mean <= ci_upper
        
        print(f"{confidence_level*100}% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
        print(f"  Contains true mean: {contains_true}")
        print(f"  Width: {ci_upper - ci_lower:.2f}")
        print()

confidence_interval_demo()
```

### 5.2 Sampling Distribution

```python
def sampling_distribution_demo():
    # Population parameters
    population_mean = 50
    population_std = 15
    sample_size = 30
    n_samples = 1000
    
    # Generate many samples and calculate their means
    sample_means = []
    for _ in range(n_samples):
        sample = np.random.normal(population_mean, population_std, sample_size)
        sample_means.append(np.mean(sample))
    
    sample_means = np.array(sample_means)
    
    # Theoretical sampling distribution properties
    theoretical_mean = population_mean
    theoretical_std = population_std / np.sqrt(sample_size)
    
    # Empirical sampling distribution properties
    empirical_mean = np.mean(sample_means)
    empirical_std = np.std(sample_means, ddof=1)
    
    print(f"Sampling Distribution of Sample Means:")
    print(f"Sample size: {sample_size}")
    print(f"Number of samples: {n_samples}")
    print()
    print(f"Theoretical mean: {theoretical_mean:.2f}")
    print(f"Empirical mean: {empirical_mean:.2f}")
    print()
    print(f"Theoretical standard error: {theoretical_std:.2f}")
    print(f"Empirical standard error: {empirical_std:.2f}")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Histogram of sample means
    plt.subplot(1, 2, 1)
    plt.hist(sample_means, bins=30, alpha=0.7, density=True, label='Empirical')
    
    # Theoretical normal distribution
    x = np.linspace(np.min(sample_means), np.max(sample_means), 100)
    theoretical_pdf = norm.pdf(x, theoretical_mean, theoretical_std)
    plt.plot(x, theoretical_pdf, 'r-', linewidth=2, label='Theoretical')
    
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.title('Sampling Distribution of Sample Means')
    plt.legend()
    
    # QQ plot to check normality
    plt.subplot(1, 2, 2)
    stats.probplot(sample_means, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Checking Normality)')
    
    plt.tight_layout()
    plt.show()

sampling_distribution_demo()
```

---

## 6. Bayesian Statistics {#bayesian}

### 6.1 Bayes' Theorem

```python
def bayes_theorem_demo():
    # Email spam detection example
    print("Email Spam Detection Example")
    print("=" * 40)
    
    # Prior probabilities
    p_spam = 0.3  # 30% of emails are spam
    p_not_spam = 0.7  # 70% of emails are not spam
    
    # Likelihoods (probability of word "free" appearing)
    p_free_given_spam = 0.8  # 80% of spam emails contain "free"
    p_free_given_not_spam = 0.1  # 10% of non-spam emails contain "free"
    
    # Calculate marginal probability P(free)
    p_free = p_free_given_spam * p_spam + p_free_given_not_spam * p_not_spam
    
    # Calculate posterior probabilities using Bayes' theorem
    p_spam_given_free = (p_free_given_spam * p_spam) / p_free
    p_not_spam_given_free = (p_free_given_not_spam * p_not_spam) / p_free
    
    print(f"Prior P(Spam) = {p_spam:.3f}")
    print(f"Prior P(Not Spam) = {p_not_spam:.3f}")
    print()
    print(f"Likelihood P(Free|Spam) = {p_free_given_spam:.3f}")
    print(f"Likelihood P(Free|Not Spam) = {p_free_given_not_spam:.3f}")
    print()
    print(f"Marginal P(Free) = {p_free:.3f}")
    print()
    print(f"Posterior P(Spam|Free) = {p_spam_given_free:.3f}")
    print(f"Posterior P(Not Spam|Free) = {p_not_spam_given_free:.3f}")
    print()
    print(f"Decision: Email is {'SPAM' if p_spam_given_free > 0.5 else 'NOT SPAM'}")

bayes_theorem_demo()
```

### 6.2 Bayesian Updating

```python
def bayesian_updating_demo():
    # Coin flip example - estimating probability of heads
    print("Bayesian Coin Flip Example")
    print("=" * 40)
    
    # Prior belief (uniform distribution)
    p_values = np.linspace(0, 1, 1000)
    prior = np.ones_like(p_values)  # Uniform prior
    prior = prior / np.sum(prior)  # Normalize
    
    # Simulate coin flips
    true_p = 0.7  # True probability of heads
    flips = np.random.binomial(1, true_p, 20)
    
    # Update belief after each flip
    posterior = prior.copy()
    n_heads = 0
    n_tails = 0
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    update_points = [0, 5, 10, 20]
    
    for i, n_flips in enumerate(update_points):
        if n_flips > 0:
            # Count heads and tails up to this point
            n_heads = np.sum(flips[:n_flips])
            n_tails = n_flips - n_heads
            
            # Update posterior using Beta distribution (conjugate prior)
            # Beta(α, β) where α = 1 + n_heads, β = 1 + n_tails
            alpha = 1 + n_heads
            beta = 1 + n_tails
            posterior = stats.beta.pdf(p_values, alpha, beta)
        
        axes[i].plot(p_values, posterior, 'b-', linewidth=2)
        axes[i].axvline(true_p, color='red', linestyle='--', label=f'True p = {true_p}')
        axes[i].set_xlabel('Probability of Heads')
        axes[i].set_ylabel('Posterior Density')
        axes[i].set_title(f'After {n_flips} flips: {n_heads} heads, {n_tails} tails')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        