# Comprehensive Algebra for Machine Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Algebraic Operations](#basic-algebraic-operations)
3. [Functions and Their Properties](#functions-and-their-properties)
4. [Linear Equations and Systems](#linear-equations-and-systems)
5. [Quadratic Functions and Equations](#quadratic-functions-and-equations)
6. [Exponential and Logarithmic Functions](#exponential-and-logarithmic-functions)
7. [Polynomials](#polynomials)
8. [Inequalities](#inequalities)
9. [Sequences and Series](#sequences-and-series)
10. [Matrix Algebra Fundamentals](#matrix-algebra-fundamentals)
11. [Optimization Concepts](#optimization-concepts)
12. [Applications in Machine Learning](#applications-in-machine-learning)

---

## Introduction

Algebra forms the mathematical backbone of machine learning. Understanding these concepts is crucial for grasping how ML algorithms work, from simple linear regression to complex neural networks. This tutorial focuses on the algebraic foundations most relevant to ML practitioners.

---

## Basic Algebraic Operations

### Variables and Constants
- **Variables**: Symbols (usually letters) that represent unknown or changing values
- **Constants**: Fixed numerical values
- **Coefficients**: Numbers multiplying variables (e.g., in 3x, the coefficient is 3)

### Order of Operations (PEMDAS/BODMAS)
1. **Parentheses/Brackets**
2. **Exponents/Orders**
3. **Multiplication and Division** (left to right)
4. **Addition and Subtraction** (left to right)

### Distributive Property
**Rule**: a(b + c) = ab + ac

**Example**: 3(x + 2) = 3x + 6

**ML Relevance**: Essential for expanding cost functions and gradient calculations.

### Combining Like Terms
**Rule**: Add/subtract coefficients of identical variable terms

**Example**: 3x + 5x - 2x = 6x

### Factoring
**Common Factor**: 6x + 9 = 3(2x + 3)
**Difference of Squares**: a² - b² = (a + b)(a - b)
**Quadratic Factoring**: x² + 5x + 6 = (x + 2)(x + 3)

---

## Functions and Their Properties

### Definition of a Function
A function f maps each input x to exactly one output y: f(x) = y

### Function Notation
- f(x) = 2x + 3
- g(t) = t² - 4t + 1
- h(z) = e^z

### Domain and Range
- **Domain**: All possible input values
- **Range**: All possible output values

### Types of Functions in ML

#### Linear Functions
**Form**: f(x) = mx + b
- m = slope
- b = y-intercept

**Example**: f(x) = 2x + 1

#### Quadratic Functions
**Form**: f(x) = ax² + bx + c
- a ≠ 0
- Creates parabola shape

#### Exponential Functions
**Form**: f(x) = aᵇˣ or f(x) = ae^(bx)

**Example**: f(x) = 2e^(3x)

#### Logarithmic Functions
**Form**: f(x) = log_a(x) or f(x) = ln(x)

### Function Operations
- **Addition**: (f + g)(x) = f(x) + g(x)
- **Multiplication**: (f · g)(x) = f(x) · g(x)
- **Composition**: (f ∘ g)(x) = f(g(x))

### Inverse Functions
If f(x) = y, then f⁻¹(y) = x

**Finding Inverse**:
1. Replace f(x) with y
2. Swap x and y
3. Solve for y
4. Replace y with f⁻¹(x)

**Example**: 
- f(x) = 2x + 3
- y = 2x + 3
- x = 2y + 3
- y = (x - 3)/2
- f⁻¹(x) = (x - 3)/2

---

## Linear Equations and Systems

### Single Linear Equations
**Standard Form**: ax + b = 0
**Solution**: x = -b/a

### Slope-Intercept Form
**Form**: y = mx + b
- m = slope = (y₂ - y₁)/(x₂ - x₁)
- b = y-intercept

### Systems of Linear Equations
**Two variables**: 
```
ax + by = c
dx + ey = f
```

### Solution Methods

#### Substitution Method
1. Solve one equation for one variable
2. Substitute into the other equation
3. Solve for remaining variable
4. Back-substitute

#### Elimination Method
1. Multiply equations to make coefficients of one variable equal
2. Add/subtract equations to eliminate that variable
3. Solve for remaining variable
4. Back-substitute

#### Matrix Method (Preview)
For system Ax = b, solution is x = A⁻¹b (when A is invertible)

### Types of Solutions
- **Unique Solution**: Lines intersect at one point
- **No Solution**: Parallel lines
- **Infinite Solutions**: Same line

---

## Quadratic Functions and Equations

### Standard Form
**Form**: ax² + bx + c = 0 (a ≠ 0)

### Quadratic Formula
**Formula**: x = (-b ± √(b² - 4ac))/(2a)

**Discriminant**: Δ = b² - 4ac
- Δ > 0: Two real solutions
- Δ = 0: One real solution
- Δ < 0: No real solutions

### Vertex Form
**Form**: f(x) = a(x - h)² + k
- Vertex at (h, k)
- h = -b/(2a)
- k = f(h)

### Completing the Square
**Process**: Convert ax² + bx + c to a(x - h)² + k

**Example**: x² + 6x + 8
1. x² + 6x + 9 - 9 + 8
2. (x + 3)² - 1

### Applications in ML
- **Cost Functions**: Often quadratic in parameters
- **Optimization**: Finding minima of quadratic functions
- **Regularization**: L2 regularization adds quadratic terms

---

## Exponential and Logarithmic Functions

### Exponential Functions
**Form**: f(x) = aˣ (a > 0, a ≠ 1)

**Properties**:
- aˣ · aʸ = aˣ⁺ʸ
- aˣ / aʸ = aˣ⁻ʸ
- (aˣ)ʸ = aˣʸ
- a⁰ = 1
- a⁻ˣ = 1/aˣ

### Natural Exponential Function
**Form**: f(x) = eˣ
- e ≈ 2.71828...
- Most important in ML

### Logarithmic Functions
**Definition**: If y = aˣ, then x = log_a(y)

**Properties**:
- log_a(xy) = log_a(x) + log_a(y)
- log_a(x/y) = log_a(x) - log_a(y)
- log_a(xⁿ) = n·log_a(x)
- log_a(1) = 0
- log_a(a) = 1

### Natural Logarithm
**Form**: ln(x) = log_e(x)

**Key Property**: ln(eˣ) = x and e^(ln(x)) = x

### Change of Base Formula
**Formula**: log_a(x) = ln(x)/ln(a)

### Applications in ML
- **Activation Functions**: Sigmoid, softmax
- **Loss Functions**: Cross-entropy loss
- **Probability**: Log-likelihood
- **Optimization**: Log-space computations for numerical stability

---

## Polynomials

### Definition
**Form**: P(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀

### Degree and Leading Coefficient
- **Degree**: Highest power of x
- **Leading Coefficient**: Coefficient of highest degree term

### Operations on Polynomials

#### Addition/Subtraction
Combine like terms:
(3x² + 2x - 1) + (x² - 4x + 3) = 4x² - 2x + 2

#### Multiplication
Use distributive property:
(x + 2)(x - 3) = x² - 3x + 2x - 6 = x² - x - 6

#### Division
**Synthetic Division** or **Long Division**

### Polynomial Equations
**Finding Roots**: Values of x where P(x) = 0

**Fundamental Theorem**: A polynomial of degree n has exactly n roots (counting multiplicities)

### Rational Root Theorem
Possible rational roots are ±(factors of constant term)/(factors of leading coefficient)

### Applications in ML
- **Feature Engineering**: Polynomial features
- **Approximation**: Taylor series
- **Regularization**: Higher-order penalty terms

---

## Inequalities

### Basic Inequality Operations
**Rules**:
1. Adding/subtracting same value preserves inequality
2. Multiplying/dividing by positive number preserves inequality
3. Multiplying/dividing by negative number reverses inequality

### Linear Inequalities
**Form**: ax + b < c (or ≤, >, ≥)

**Solution**: x < (c - b)/a (if a > 0)

### Systems of Inequalities
**Graphical Solution**: Find intersection of half-planes

### Quadratic Inequalities
**Process**:
1. Find roots of corresponding equation
2. Test intervals between roots
3. Determine sign of quadratic in each interval

### Absolute Value Inequalities
**Type 1**: |x| < a means -a < x < a
**Type 2**: |x| > a means x < -a or x > a

### Applications in ML
- **Constraints**: Optimization problems
- **Regularization**: L1 regularization involves absolute values
- **Support Vector Machines**: Inequality constraints

---

## Sequences and Series

### Sequences
**Definition**: Ordered list of numbers

**Notation**: {aₙ} or a₁, a₂, a₃, ...

### Arithmetic Sequences
**Form**: aₙ = a₁ + (n-1)d
- d = common difference
- **Sum**: Sₙ = n/2[2a₁ + (n-1)d]

### Geometric Sequences
**Form**: aₙ = a₁ · rⁿ⁻¹
- r = common ratio
- **Sum**: Sₙ = a₁(1 - rⁿ)/(1 - r) if r ≠ 1

### Infinite Series
**Geometric Series**: ∑(a₁ · rⁿ⁻¹) = a₁/(1-r) if |r| < 1

### Applications in ML
- **Iterative Algorithms**: Gradient descent
- **Convergence**: Algorithm convergence rates
- **Regularization**: Series expansions

---

## Matrix Algebra Fundamentals

### Matrix Definition
**Matrix**: Rectangular array of numbers

**Notation**: A = [aᵢⱼ] where i = row, j = column

### Matrix Operations

#### Addition/Subtraction
**Rule**: Add/subtract corresponding elements
**Requirement**: Same dimensions

#### Scalar Multiplication
**Rule**: Multiply each element by scalar

#### Matrix Multiplication
**Rule**: (AB)ᵢⱼ = ∑ₖ(aᵢₖ · bₖⱼ)
**Requirement**: Columns of A = Rows of B

### Special Matrices
- **Identity Matrix**: I (diagonal of 1s)
- **Zero Matrix**: O (all zeros)
- **Transpose**: Aᵀ (swap rows and columns)

### Matrix Properties
- **Associative**: (AB)C = A(BC)
- **Distributive**: A(B + C) = AB + AC
- **NOT Commutative**: AB ≠ BA (generally)

### Determinant (2×2)
**Formula**: det(A) = ad - bc for A = [a b; c d]

### Inverse Matrix
**Definition**: A⁻¹ such that AA⁻¹ = I

**2×2 Inverse**: A⁻¹ = (1/det(A))[d -b; -c a]

### Applications in ML
- **Linear Transformations**: Feature transformations
- **Systems**: Solving linear systems
- **Principal Component Analysis**: Eigenvalues and eigenvectors
- **Neural Networks**: Weight matrices

---

## Optimization Concepts

### Function Optimization
**Goal**: Find x that maximizes or minimizes f(x)

### Critical Points
**Definition**: Points where f'(x) = 0 or f'(x) is undefined

### Local vs Global Extrema
- **Local Maximum**: Highest point in neighborhood
- **Global Maximum**: Highest point overall
- **Local Minimum**: Lowest point in neighborhood
- **Global Minimum**: Lowest point overall

### Optimization for Quadratic Functions
**Form**: f(x) = ax² + bx + c

**Vertex**: x = -b/(2a)
- If a > 0: minimum at vertex
- If a < 0: maximum at vertex

### Constrained Optimization (Preview)
**Problem**: Optimize f(x) subject to g(x) = 0

**Method**: Lagrange multipliers

### Applications in ML
- **Cost Function Minimization**: Training algorithms
- **Gradient Descent**: Iterative optimization
- **Regularization**: Balancing fit and complexity

---

## Applications in Machine Learning

### Linear Regression
**Model**: y = mx + b
**Cost Function**: J(m,b) = (1/2n)∑(yᵢ - mxᵢ - b)²

### Polynomial Regression
**Model**: y = a₀ + a₁x + a₂x² + ... + aₙxⁿ

### Logistic Regression
**Model**: p = 1/(1 + e⁻ᶻ) where z = mx + b
**Uses**: Exponential and logarithmic functions

### Neural Networks
**Linear Combinations**: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
**Activation**: a = f(z) where f might be sigmoid, tanh, ReLU

### Regularization
**L1 (Lasso)**: Adds ∑|wᵢ| to cost function
**L2 (Ridge)**: Adds ∑wᵢ² to cost function

### Gradient Descent
**Update Rule**: w := w - α∇J(w)
**Requires**: Understanding of functions and optimization

### Principal Component Analysis
**Involves**: Matrix operations, eigenvalues, eigenvectors

### Support Vector Machines
**Uses**: Linear algebra, optimization with constraints

---

## Practice Problems

### Basic Operations
1. Simplify: 3(2x - 1) + 2(x + 3)
2. Factor: x² - 9
3. Solve: 2x + 5 = 13

### Functions
1. Find f(3) if f(x) = x² - 2x + 1
2. Find the inverse of f(x) = 3x - 2
3. Simplify: log₂(8) + log₂(4)

### Systems
1. Solve: 2x + y = 7, x - y = 2
2. Find intersection of y = x² and y = 2x + 3

### Quadratics
1. Solve: x² - 5x + 6 = 0
2. Find vertex of f(x) = x² - 4x + 3
3. Complete the square: x² + 8x + 12

### Exponentials
1. Solve: 2^x = 32
2. Simplify: e^(ln(5))
3. Evaluate: ln(e³)

### Matrices
1. Multiply: [1 2; 3 4][2 1; 0 3]
2. Find determinant: [3 2; 1 4]
3. Find inverse: [2 1; 3 2]

---

## Solutions to Practice Problems

### Basic Operations
1. 3(2x - 1) + 2(x + 3) = 6x - 3 + 2x + 6 = 8x + 3
2. x² - 9 = (x + 3)(x - 3)
3. 2x + 5 = 13 → 2x = 8 → x = 4

### Functions
1. f(3) = 3² - 2(3) + 1 = 9 - 6 + 1 = 4
2. y = 3x - 2 → x = 3y - 2 → y = (x + 2)/3 → f⁻¹(x) = (x + 2)/3
3. log₂(8) + log₂(4) = log₂(8 × 4) = log₂(32) = 5

### Systems
1. From second equation: x = y + 2. Substitute: 2(y + 2) + y = 7 → 3y + 4 = 7 → y = 1, x = 3
2. x² = 2x + 3 → x² - 2x - 3 = 0 → (x - 3)(x + 1) = 0 → x = 3 or x = -1

### Quadratics
1. x² - 5x + 6 = 0 → (x - 2)(x - 3) = 0 → x = 2 or x = 3
2. x = -(-4)/(2·1) = 2, y = 2² - 4(2) + 3 = -1. Vertex: (2, -1)
3. x² + 8x + 12 = x² + 8x + 16 - 16 + 12 = (x + 4)² - 4

### Exponentials
1. 2^x = 32 = 2⁵ → x = 5
2. e^(ln(5)) = 5
3. ln(e³) = 3

### Matrices
1. [1 2; 3 4][2 1; 0 3] = [2 7; 6 15]
2. det([3 2; 1 4]) = 3(4) - 2(1) = 12 - 2 = 10
3. det = 2(2) - 1(3) = 1, so inverse = [2 -1; -3 2]

---

## Next Steps

This tutorial covers the essential algebraic foundations for machine learning. To deepen your understanding:

1. **Practice regularly** with the provided problems
2. **Connect concepts** to specific ML algorithms
3. **Study linear algebra** for advanced ML topics
4. **Learn calculus** for optimization and gradients
5. **Apply knowledge** to real ML projects

Remember: algebra is the language of machine learning. Master these concepts, and you'll find ML algorithms much more intuitive and approachable.