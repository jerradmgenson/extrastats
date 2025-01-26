# extrastats

**Advanced statistical tools and routines for Python**

`extrastats` is a Python library that provides high-quality statistical methods to address gaps in mainstream libraries like NumPy, SciPy, and statsmodels. Designed for data scientists, statisticians, and researchers, `extrastats` includes robust, customizable, and performance-optimized routines.

Copyright 2022-2025 Jerrad Michael Genson

This library is licensed under the Mozilla Public License, v. 2.0.

---

## Features

### Robust Outlier Detection
- **Adjusted Boxplot**: A method that extends traditional boxplots using the medcouple statistic for skewness adjustment.

### Advanced Permutation Testing
- Flexible resampling strategies:
  - Pairings, samples, independent shuffles, and bootstrapping.
- Parallel computation with `joblib`.

### Confidence Interval Estimation
- Bootstrap-based confidence intervals for arbitrary statistics.
- Support for multiple confidence levels in a single call.

### Sample Size Estimation
- Monte Carlo simulations to determine required sample sizes for target confidence interval widths.

### Tail Weight Analysis
- Evaluate the tail weight of distributions using the L/RMC method.

### Mutual Information
- Compute mutual information for discrete variables.
- Optional normalization for interpretability.

### Additional Metrics
- Geometric Coefficient of Variation (GCV)
- Harmonic Variability (HVAR)
- Trimmed statistics and probability operations.

---
