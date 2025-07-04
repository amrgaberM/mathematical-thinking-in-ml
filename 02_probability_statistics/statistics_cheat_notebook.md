# Statistics Cheat Notebook 📊

*A comprehensive guide to statistical concepts with practical Python implementations*

---

## Table of Contents

1. [What Is Data?](#1-what-is-data)
2. [Descriptive Versus Inferential Statistics](#2-descriptive-versus-inferential-statistics)
3. [Populations, Samples, and Bias](#3-populations-samples-and-bias)
4. [Descriptive Statistics](#4-descriptive-statistics)
   - [Mean and Weighted Mean](#41-mean-and-weighted-mean)
   - [Median](#42-median)
   - [Mode](#43-mode)
   - [Variance and Standard Deviation](#44-variance-and-standard-deviation)
   - [The Normal Distribution](#45-the-normal-distribution)
   - [The Inverse CDF](#46-the-inverse-cdf)
   - [Z-Scores](#47-z-scores)
5. [Inferential Statistics](#5-inferential-statistics)
   - [The Central Limit Theorem](#51-the-central-limit-theorem)
   - [Confidence Intervals](#52-confidence-intervals)
   - [Understanding P-Values](#53-understanding-p-values)
   - [Hypothesis Testing](#54-hypothesis-testing)
6. [The T-Distribution: Dealing with Small Samples](#6-the-t-distribution-dealing-with-small-samples)
7. [Big Data Considerations and the Texas Sharpshooter Fallacy](#7-big-data-considerations-and-the-texas-sharpshooter-fallacy)
8. [Conclusion](#8-conclusion)
9. [Exercises](#9-exercises)

---

## 1. What Is Data?

**Data** is information that can be analyzed to reveal patterns, trends, and insights. Understanding the types of data is crucial for choosing appropriate statistical methods.

### Types of Data

#### Quantitative Data (Numerical)
- **Discrete**: Countable values with gaps between possible values
  - Examples: Number of students (25, 30, 42), Number of cars, Population count
- **Continuous**: Measurable values that can take any value within a range
  - Examples: Height (5.7 feet), Weight (150.5 lbs), Temperature, Time

#### Qualitative Data (Categorical)
- **Nominal**: Categories without natural order
  - Examples: Colors (red, blue, green), Gender, Nationality
- **Ordinal**: Categories with natural order
  - Examples: Ratings (poor, fair, good, excellent), Education levels, Survey responses

### 💻 Code Example: Identifying Data Types

```python
import pandas as pd
import numpy as np

# Create sample dataset
data = {
    'student_id': [1, 2, 3, 4, 5],           # Discrete quantitative
    'height': [5.6, 5.9, 6.1, 5.4, 5.8],    # Continuous quantitative
    'grade': ['A', 'B', 'A', 'C', 'B'],      # Ordinal qualitative
    'favorite_color': ['red', 'blue', 'green', 'red', 'blue']  # Nominal
}

df = pd.DataFrame(data)
print("Data types:")
print(df.dtypes)
print("\nSample data:")
print(df.head())

# Check for data type characteristics
print(f"\nUnique values in grade: {df['grade'].unique()}")
print(f"Range of heights: {df['height'].min()} to {df['height'].max()}")
```

---

## 2. Descriptive Versus Inferential Statistics

### Descriptive Statistics

**Descriptive Statistics** summarize and describe the main features of a dataset without making predictions or generalizations beyond the data at hand.

- **Purpose**: What happened in the data?
- **Common Methods**:
  - Measures of central tendency (mean, median, mode)
  - Measures of variability (variance, standard deviation, range)
  - Graphical representations (histograms, box plots, scatter plots)
  - Frequency distributions

### Inferential Statistics

**Inferential Statistics** use sample data to make predictions, test hypotheses, and draw conclusions about a larger population.

- **Purpose**: What can we predict or conclude about the population?
- **Common Methods**:
  - Hypothesis testing
  - Confidence intervals
  - Regression analysis
  - ANOVA (Analysis of Variance)

### 💻 Code Example: Descriptive vs Inferential Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Generate sample data
np.random.seed(42)
sample_scores = np.random.normal(75, 10, 50)  # 50 test scores

# DESCRIPTIVE STATISTICS
print("=== DESCRIPTIVE STATISTICS ===")
print(f"Mean: {np.mean(sample_scores):.2f}")
print(f"Median: {np.median(sample_scores):.2f}")
print(f"Standard Deviation: {np.std(sample_scores, ddof=1):.2f}")
print(f"Min: {np.min(sample_scores):.2f}")
print(f"Max: {np.max(sample_scores):.2f}")

# INFERENTIAL STATISTICS
print("\n=== INFERENTIAL STATISTICS ===")
# 95% confidence interval for population mean
confidence_interval = stats.t.interval(0.95, len(sample_scores)-1, 
                                      loc=np.mean(sample_scores), 
                                      scale=stats.sem(sample_scores))
print(f"95% Confidence Interval: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")

# Hypothesis test: Is the population mean different from 70?
t_stat, p_value = stats.ttest_1samp(sample_scores, 70)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
```

---

## 3. Populations, Samples, and Bias

### Key Concepts

- **Population**: The entire group of individuals or items that we want to study
- **Sample**: A subset of the population used for analysis
- **Parameter**: A numerical summary of a population (typically denoted with Greek letters: μ, σ)
- **Statistic**: A numerical summary of a sample (typically denoted with Latin letters: x̄, s)

### Types of Bias

> ⚠️ **Important**: Bias can significantly affect the validity of statistical conclusions.

Common types of bias include:

1. **Selection Bias**: Non-representative sampling methods
2. **Confirmation Bias**: Seeking information that confirms preconceptions
3. **Survivorship Bias**: Focusing only on successful outcomes while ignoring failures
4. **Response Bias**: Participants provide inaccurate or misleading responses
5. **Sampling Bias**: Systematic error in sample selection

### 💻 Code Example: Sampling Methods and Bias

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a population (simulated university students)
np.random.seed(42)
population_size = 10000
population_gpa = np.random.normal(3.2, 0.5, population_size)
population_gpa = np.clip(population_gpa, 0, 4)  # Clip to valid GPA range

print(f"Population mean GPA: {np.mean(population_gpa):.3f}")
print(f"Population std GPA: {np.std(population_gpa):.3f}")

# Simple Random Sampling
sample_size = 100
random_sample = np.random.choice(population_gpa, sample_size, replace=False)
print(f"\nRandom sample mean: {np.mean(random_sample):.3f}")

# Biased Sampling (only high-performing students)
biased_sample = population_gpa[population_gpa > 3.5][:sample_size]
print(f"Biased sample mean: {np.mean(biased_sample):.3f}")

# Stratified Sampling
# Divide population into strata and sample from each
low_gpa = population_gpa[population_gpa < 2.5]
mid_gpa = population_gpa[(population_gpa >= 2.5) & (population_gpa < 3.5)]
high_gpa = population_gpa[population_gpa >= 3.5]

stratified_sample = np.concatenate([
    np.random.choice(low_gpa, 20, replace=False),
    np.random.choice(mid_gpa, 60, replace=False),
    np.random.choice(high_gpa, 20, replace=False)
])
print(f"Stratified sample mean: {np.mean(stratified_sample):.3f}")
```

---

## 4. Descriptive Statistics

Descriptive statistics provide a summary of the main characteristics of a dataset through measures of central tendency, variability, and distribution shape.

### 4.1 Mean and Weighted Mean

#### Arithmetic Mean
The arithmetic mean (average) is calculated as:

**Formula**: `x̄ = (x₁ + x₂ + ... + xₙ) / n`

#### Weighted Mean
When observations have different weights or importance:

**Formula**: `x̄ᵨ = (Σ wᵢxᵢ) / (Σ wᵢ)`

where `wᵢ` is the weight for observation `xᵢ`.

### 💻 Code Example: Mean and Weighted Mean

```python
import numpy as np

# Sample data: test scores
scores = [85, 92, 78, 96, 88, 79, 93, 87]

# Arithmetic mean
mean_score = np.mean(scores)
print(f"Arithmetic mean: {mean_score:.2f}")

# Weighted mean example: different tests have different weights
test_weights = [0.2, 0.2, 0.15, 0.25, 0.1, 0.05, 0.03, 0.02]
weighted_mean = np.average(scores, weights=test_weights)
print(f"Weighted mean: {weighted_mean:.2f}")

# Manual calculation for verification
manual_weighted = sum(s * w for s, w in zip(scores, test_weights)) / sum(test_weights)
print(f"Manual weighted mean: {manual_weighted:.2f}")
```

### 4.2 Median

The **median** is the middle value when data is arranged in ascending order. For an even number of observations, it's the average of the two middle values.

**Formula**:
- If n is odd: Median = x₍ₙ₊₁₎/₂
- If n is even: Median = (xₙ/₂ + x₍ₙ/₂₎₊₁) / 2

### 4.3 Mode

The **mode** is the value that appears most frequently in a dataset. A dataset can have:
- No mode (all values appear with equal frequency)
- One mode (unimodal)
- Two modes (bimodal)
- Multiple modes (multimodal)

### 💻 Code Example: Median and Mode

```python
import numpy as np
from scipy import stats

# Sample data
data = [1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 10]

# Median
median_value = np.median(data)
print(f"Median: {median_value}")

# Mode
mode_result = stats.mode(data, keepdims=True)
print(f"Mode: {mode_result.mode[0]} (appears {mode_result.count[0]} times)")

# For comparison with mean
mean_value = np.mean(data)
print(f"Mean: {mean_value:.2f}")
print(f"Median: {median_value}")
print(f"Mode: {mode_result.mode[0]}")

# Demonstrate with skewed data
skewed_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
print(f"\nSkewed data:")
print(f"Mean: {np.mean(skewed_data):.2f}")
print(f"Median: {np.median(skewed_data):.2f}")
print("Notice how the outlier (100) affects the mean but not the median")
```

### 4.4 Variance and Standard Deviation

**Variance** measures the average squared deviation from the mean, while **standard deviation** is the square root of variance, expressed in the same units as the original data.

**Formulas**:
- **Population Variance**: σ² = Σ(xᵢ - μ)² / N
- **Sample Variance**: s² = Σ(xᵢ - x̄)² / (n-1)
- **Standard Deviation**: σ = √σ² (population), s = √s² (sample)

### 4.5 The Normal Distribution

The **normal distribution** is a continuous probability distribution characterized by its bell-shaped curve. It's defined by two parameters: mean (μ) and standard deviation (σ).

**Key Properties**:
- Symmetric around the mean
- Mean = Median = Mode
- 68% of data within 1 standard deviation
- 95% of data within 2 standard deviations
- 99.7% of data within 3 standard deviations (68-95-99.7 rule)

### 4.6 The Inverse CDF

The **Inverse Cumulative Distribution Function (Inverse CDF)** or **quantile function** gives the value below which a certain percentage of observations fall.

### 4.7 Z-Scores

A **Z-score** (standard score) indicates how many standard deviations an element is from the mean.

**Formula**: `z = (x - μ) / σ` for population, or `z = (x - x̄) / s` for sample

### 💻 Code Example: Variance, Standard Deviation, and Z-Scores

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Sample data
data = [85, 92, 78, 96, 88, 79, 93, 87, 90, 82]

# Variance and Standard Deviation
sample_var = np.var(data, ddof=1)  # ddof=1 for sample variance
sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation

print(f"Sample variance: {sample_var:.2f}")
print(f"Sample standard deviation: {sample_std:.2f}")

# Z-scores
mean_data = np.mean(data)
z_scores = [(x - mean_data) / sample_std for x in data]
print(f"\nZ-scores: {[round(z, 2) for z in z_scores]}")

# Using scipy for z-scores
z_scores_scipy = stats.zscore(data)
print(f"Z-scores (scipy): {[round(z, 2) for z in z_scores_scipy]}")

# Normal distribution example
np.random.seed(42)
normal_data = np.random.normal(100, 15, 1000)  # mean=100, std=15

# Calculate percentiles using inverse CDF
percentiles = [25, 50, 75, 90, 95, 99]
values = np.percentile(normal_data, percentiles)

print(f"\nPercentiles of normal data:")
for p, v in zip(percentiles, values):
    print(f"{p}th percentile: {v:.2f}")

# Demonstrate 68-95-99.7 rule
within_1_std = np.sum(np.abs(stats.zscore(normal_data)) <= 1) / len(normal_data)
within_2_std = np.sum(np.abs(stats.zscore(normal_data)) <= 2) / len(normal_data)
within_3_std = np.sum(np.abs(stats.zscore(normal_data)) <= 3) / len(normal_data)

print(f"\nEmpirical Rule verification:")
print(f"Within 1 std: {within_1_std:.1%} (expected: 68%)")
print(f"Within 2 std: {within_2_std:.1%} (expected: 95%)")
print(f"Within 3 std: {within_3_std:.1%} (expected: 99.7%)")
```

---

## 5. Inferential Statistics

Inferential statistics allow us to make predictions and draw conclusions about populations based on sample data.

### 5.1 The Central Limit Theorem

The **Central Limit Theorem (CLT)** states that the distribution of sample means approaches a normal distribution as the sample size increases, regardless of the shape of the population distribution.

**Key Points**:
- Sample size should be at least 30 for CLT to apply effectively
- Mean of sample means equals population mean: μₓ̄ = μ
- Standard deviation of sample means (standard error): σₓ̄ = σ/√n

### 5.2 Confidence Intervals

A **confidence interval** provides a range of values that likely contains the true population parameter with a specified level of confidence.

**Formulas**:
- For known standard deviation: `x̄ ± z(α/2) × (σ/√n)`
- For unknown standard deviation: `x̄ ± t(α/2) × (s/√n)`

### 5.3 Understanding P-Values

A **p-value** is the probability of observing test results at least as extreme as the observed results, assuming the null hypothesis is true.

**Interpretation**:
- p < 0.05: Strong evidence against null hypothesis
- p < 0.01: Very strong evidence against null hypothesis
- p ≥ 0.05: Insufficient evidence to reject null hypothesis

### 5.4 Hypothesis Testing

**Hypothesis testing** is a statistical method used to make inferences about population parameters based on sample data.

**Steps**:
1. State null hypothesis (H₀) and alternative hypothesis (H₁)
2. Choose significance level (α)
3. Calculate test statistic
4. Determine p-value
5. Make decision (reject or fail to reject H₀)

### 💻 Code Example: Inferential Statistics

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Demonstrate Central Limit Theorem
np.random.seed(42)

# Create a non-normal population (exponential distribution)
population = np.random.exponential(scale=2, size=10000)

# Take many samples and calculate their means
sample_size = 30
num_samples = 1000
sample_means = []

for _ in range(num_samples):
    sample = np.random.choice(population, sample_size, replace=False)
    sample_means.append(np.mean(sample))

# Check if sample means are normally distributed
print("Central Limit Theorem Demonstration:")
print(f"Population mean: {np.mean(population):.3f}")
print(f"Mean of sample means: {np.mean(sample_means):.3f}")
print(f"Population std: {np.std(population):.3f}")
print(f"Std of sample means: {np.std(sample_means):.3f}")
print(f"Expected std of sample means: {np.std(population)/np.sqrt(sample_size):.3f}")

# Confidence Interval Example
sample_data = np.random.normal(100, 15, 50)
confidence_level = 0.95
alpha = 1 - confidence_level

# Calculate confidence interval
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1)
standard_error = sample_std / np.sqrt(len(sample_data))

# Using t-distribution for small sample
t_value = stats.t.ppf(1 - alpha/2, df=len(sample_data)-1)
margin_of_error = t_value * standard_error

ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print(f"\n95% Confidence Interval:")
print(f"Sample mean: {sample_mean:.2f}")
print(f"Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")

# Hypothesis Testing Example
# H0: μ = 100 vs H1: μ ≠ 100
hypothesized_mean = 100
t_statistic = (sample_mean - hypothesized_mean) / standard_error
p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=len(sample_data)-1))

print(f"\nHypothesis Test (H0: μ = 100):")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Decision: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'}")
```

---

## 6. The T-Distribution: Dealing with Small Samples

The **t-distribution** is used when the population standard deviation is unknown and the sample size is small (typically n < 30).

**Key Properties**:
- Similar to normal distribution but with heavier tails
- Approaches normal distribution as degrees of freedom increase
- Degrees of freedom = n - 1
- More conservative than z-distribution (wider confidence intervals)

**Formulas**:
- T-statistic for one sample: `t = (x̄ - μ₀) / (s/√n)`
- T-statistic for two samples: `t = (x̄₁ - x̄₂) / √((s₁²/n₁) + (s₂²/n₂))`

### 💻 Code Example: T-Distribution Applications

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Small sample example
np.random.seed(42)
small_sample = np.random.normal(50, 10, 12)  # n = 12

# Calculate statistics
sample_mean = np.mean(small_sample)
sample_std = np.std(small_sample, ddof=1)
n = len(small_sample)
standard_error = sample_std / np.sqrt(n)

print(f"Small sample statistics (n={n}):")
print(f"Sample mean: {sample_mean:.2f}")
print(f"Sample std: {sample_std:.2f}")
print(f"Standard error: {standard_error:.2f}")

# Compare t-distribution vs normal distribution confidence intervals
confidence_level = 0.95
alpha = 1 - confidence_level

# T-distribution (appropriate for small samples)
t_value = stats.t.ppf(1 - alpha/2, df=n-1)
t_margin_error = t_value * standard_error
t_ci = (sample_mean - t_margin_error, sample_mean + t_margin_error)

# Normal distribution (inappropriate for small samples)
z_value = stats.norm.ppf(1 - alpha/2)
z_margin_error = z_value * standard_error
z_ci = (sample_mean - z_margin_error, sample_mean + z_margin_error)

print(f"\n95% Confidence Intervals:")
print(f"T-distribution: ({t_ci[0]:.2f}, {t_ci[1]:.2f})")
print(f"Normal distribution: ({z_ci[0]:.2f}, {z_ci[1]:.2f})")
print(f"T-interval is {'wider' if t_ci[1]-t_ci[0] > z_ci[1]-z_ci[0] else 'narrower'}")

# Two-sample t-test
group1 = np.random.normal(50, 10, 15)
group2 = np.random.normal(53, 10, 12)

# Perform independent t-test
t_stat, p_val = stats.ttest_ind(group1, group2)
print(f"\nTwo-sample t-test:")
print(f"Group 1 mean: {np.mean(group1):.2f}")
print(f"Group 2 mean: {np.mean(group2):.2f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")
print(f"Result: {'Significant difference' if p_val < 0.05 else 'No significant difference'}")
```

---

## 7. Big Data Considerations and the Texas Sharpshooter Fallacy

### Big Data Considerations

With large datasets, several statistical considerations become important:

> ⚠️ **Multiple Comparisons Problem**: When conducting many statistical tests, the probability of finding at least one significant result by chance alone increases dramatically.

**Key Issues**:
- **Statistical vs Practical Significance**: Large samples can make trivial differences statistically significant
- **Data Dredging**: Searching through data for patterns without proper hypothesis formation
- **Overfitting**: Models that perform well on training data but poorly on new data
- **Computational Complexity**: Processing and analyzing massive datasets requires efficient algorithms
- **Selection Bias**: Large datasets may not represent the target population

### The Texas Sharpshooter Fallacy

The **Texas Sharpshooter Fallacy** occurs when someone shoots at a barn wall and then draws a target around the bullet holes, claiming to be a sharpshooter. In statistics, this represents cherry-picking data to support a conclusion.

**How it Manifests**:
- Data mining without proper hypothesis formation
- Post-hoc analysis without correction for multiple comparisons
- Ignoring negative results while highlighting positive ones
- Clustering random data and claiming patterns exist

**Prevention Strategies**:
- Pre-register hypotheses before data collection
- Use appropriate corrections for multiple comparisons (Bonferroni, FDR)
- Cross-validation and out-of-sample testing
- Replication studies

### 💻 Code Example: Multiple Comparisons Problem

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Simulate multiple comparisons problem
np.random.seed(42)
num_tests = 100
alpha = 0.05

# Generate random data (no real effect)
p_values = []
for i in range(num_tests):
    group1 = np.random.normal(0, 1, 50)
    group2 = np.random.normal(0, 1, 50)  # Same population
    _, p_val = stats.ttest_ind(group1, group2)
    p_values.append(p_val)

# Count significant results
significant_results = sum(1 for p in p_values if p < alpha)
expected_false_positives = num_tests * alpha

print(f"Multiple Comparisons Problem Demonstration:")
print(f"Number of tests: {num_tests}")
print(f"Alpha level: {alpha}")
print(f"Expected false positives: {expected_false_positives}")
print(f"Actual significant results: {significant_results}")
print(f"Percentage of false positives: {significant_results/num_tests:.1%}")

# Bonferroni correction
bonferroni_alpha = alpha / num_tests
bonferroni_significant = sum(1 for p in p_values if p < bonferroni_alpha)
print(f"\nWith Bonferroni correction:")
print(f"Corrected alpha: {bonferroni_alpha:.6f}")
print(f"Significant results: {bonferroni_significant}")

# False Discovery Rate (FDR) correction
from statsmodels.stats.multitest import multipletests
rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
fdr_significant = sum(rejected)
print(f"\nWith FDR correction:")
print(f"Significant results: {fdr_significant}")
```

---

## 8. Conclusion

This statistics cheat notebook covers the fundamental concepts needed for data analysis and statistical inference. Key takeaways include:

### Essential Concepts
- **Data Types**: Understanding quantitative vs qualitative data guides method selection
- **Descriptive Statistics**: Summarize data characteristics through central tendency and variability measures
- **Inferential Statistics**: Make population conclusions from sample data using confidence intervals and hypothesis testing
- **Distribution Theory**: Normal distribution and Central Limit Theorem form the foundation of many statistical methods

### Best Practices
- Always visualize your data before analysis
- Check assumptions before applying statistical tests
- Be aware of bias in data collection and analysis
- Use appropriate corrections for multiple comparisons
- Consider both statistical and practical significance
- Validate results through replication and cross-validation

### Python Libraries for Statistics
- **NumPy**: Basic mathematical operations and array handling
- **SciPy**: Statistical functions and probability distributions
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Statsmodels**: Advanced statistical modeling

---

## 9. Exercises

### Exercise 1: Data Type Classification
Given the following variables, classify each as quantitative (discrete/continuous) or qualitative (nominal/ordinal):
- Number of siblings
- Customer satisfaction rating (1-5 scale)
- Blood type (A, B, AB, O)
- Time to complete a task
- ZIP code

### Exercise 2: Descriptive Statistics
Using the dataset `[23, 45, 67, 89, 12, 34, 56, 78, 90, 21]`:
1. Calculate mean, median, and mode
2. Calculate variance and standard deviation
3. Find the z-score for the value 67
4. Determine if any values are outliers (z-score > 2 or < -2)

### Exercise 3: Confidence Intervals
A sample of 25 students has a mean test score of 78 with a standard deviation of 12.
1. Calculate a 95% confidence interval for the population mean
2. Interpret the confidence interval
3. What would happen to the interval width if the sample size increased to 100?

### Exercise 4: Hypothesis Testing
A manufacturer claims their light bulbs last 1000 hours on average. A sample of 30 bulbs has a mean lifetime of 980 hours with a standard deviation of 50 hours.
1. Set up null and alternative hypotheses
2. Calculate the test statistic
3. Determine the p-value
4. Make a decision at α = 0.05

### Exercise 5: Multiple Comparisons
You're testing 20 different marketing strategies, each compared to a control group. If you use α = 0.05 for each test:
1. What's the probability of at least one Type I error?
2. What should the Bonferroni-corrected α be?
3. Discuss the trade-off between Type I and Type II errors

---

## 📚 Additional Resources

### Books
- "Think Stats" by Allen B. Downey
- "Practical Statistics for Data Scientists" by Peter Bruce and Andrew Bruce
- "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani

### Online Resources
- Khan Academy Statistics Course
- Coursera Statistical Inference Course
- Python for Data Science Handbook (online)

### Python Documentation
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Statistical Functions](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

*This notebook serves as a comprehensive reference for statistical concepts and their practical implementation in Python. For the most current information and advanced topics, refer to the official documentation and academic resources.*