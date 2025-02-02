"""
extrastats: A library of high-quality statistical routines for Python

extrastats provides advanced statistical tools and routines that fill gaps in mainstream Python packages like NumPy, SciPy, and statsmodels. These tools are optimized for accuracy, flexibility, and performance, with support for parallel computation.

Features include:
- Adjusted boxplot methods for robust outlier detection.
- Permutation tests with advanced resampling strategies.
- Confidence interval estimation via bootstrapping.
- Sample size estimation for target confidence interval widths.
- Tail weight analysis using L/RMC methods.
- Mutual information computation with optional normalization.
- Harmonic variability, geometric coefficient of variation, and more.

Key Benefits:
- Integration with popular Python libraries (e.g., NumPy, SciPy).
- Parallel execution support using `joblib`.
- Customizable randomization and resampling techniques.
- Extensive error handling and input validation.

Copyright 2022-2025 Jerrad Michael Genson

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, you can obtain one at https://mozilla.org/MPL/2.0/.

"""

import logging
import math
import warnings
from collections import namedtuple
from functools import partial, reduce, singledispatch, wraps
from itertools import batched, chain
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from scipy import stats
from statsmodels.stats.stattools import medcouple

__all__ = [
    "MedcoupleError",
    "TestResult",
    "adjusted_boxplot",
    "permutation_test",
    "confidence_interval",
    "sample_size",
    "tail_weight",
    "standard_error",
    "gcv",
    "hvar",
    "xtrim",
    "ftrim",
    "sum_prob",
    "mutual_info",
]

DEFAULT_THRESHOLD = 1.5
MAX_INT = 2147483648

TestResult = namedtuple("TestResult", "statistic pvalue")


class MedcoupleError(Exception):
    """
    Raised when there is an error in the medcouple calculation.

    """


def adjusted_boxplot(
    x,
    k=DEFAULT_THRESHOLD,
    frac=1,
    n_jobs=1,
    parallel=None,
    random_state=None,
    raise_medcouple_error=True,
):
    """
    Apply the adjusted boxplot method on an array of numeric values.

    Args:
      x: A 1-D ndarray of numeric values.
      k: Factor for calculating outlier thresholds.
         Default value is 1.5.
      frac: Fraction of the data to use for calculating the medcouple.
            When set to 1, the entire array is used.
      n_jobs: Not used in this variant of adjusted_boxplot.
      parallel: Not used in this variant of adjusted_boxplot.
      random_state: Either an integer >= 0 or an instance of
                    numpy.random.Generator. Used to attain reproducible
                    behavior when frac < 1.
      raise_medcouple_error: Raise a MedcoupleError exception when the medcouple
                             calculation results in a NaN value. Otherwise, issue
                             a warning and use the ordinary boxplot method without
                             the medcouple adjustment.

    Returns:
      A tuple of (low, high) outlier thresholds. If 'k' is a sequence,
      a generator of tuples is returned instead.

    """

    if x.ndim > 2:
        raise ValueError("adjusted_boxplot called with x.ndim > 2")

    x0 = x[~np.isnan(x)]
    if isinstance(random_state, int):
        rng = np.random.default_rng(seed=random_state)

    elif isinstance(random_state, np.random.Generator):
        rng = random_state

    elif random_state is None:
        rng = np.random.default_rng()

    else:
        raise ValueError(f"Got unexpected value for random_state: {random_state}")

    if frac < 1:
        x0 = rng.choice(x0, size=int(round(frac * len(x))), replace=False)

    q1, q3 = np.quantile(x0, [0.25, 0.75])
    iqr_ = q3 - q1
    mc = medcouple(x0)
    if np.isnan(mc):
        error = "medcouple calculation resulted in nan"
        if raise_medcouple_error:
            raise MedcoupleError(error)

        logger = logging.getLogger(__name__)
        logger.warning(error)

    def apply_threshold(k):
        if np.isnan(mc):
            threshold = k * iqr_
            lower_fence = q1 - threshold
            upper_fence = q3 + threshold

        elif mc >= 0:
            lower_fence = q1 - k * np.exp(-3.5 * mc) * iqr_
            upper_fence = q3 + k * np.exp(4 * mc) * iqr_

        else:
            lower_fence = q1 - k * np.exp(-4 * mc) * iqr_
            upper_fence = q3 + k * np.exp(3.5 * mc) * iqr_

        return lower_fence, upper_fence

    try:
        outlier_groups = (apply_threshold(t) for t in k)
        return outlier_groups

    except TypeError:
        return apply_threshold(k)


def permutation_test(
    f,
    a,
    *args,
    alternative="two-sided",
    permutation_type="bootstrap",
    less_is_more=False,
    iterations=1000,
    batch=False,
    n_jobs=1,
    parallel=None,
    random_state=None,
):
    """
    Conduct a randomized permutation test on the given datasets.

    Unlike the scipy implementation, extrastats supports parallel
    execution via joblib and bootstrap permutations.

    Args:
      f: A function that calculates the statistic of interest. Normally,
         it should accept a single ndarray and return a scalar value for
         the statistic.
      a: An ndarray representing the first dataset under study.
      *args: ndarrays representing subsequent datasets in the study.
      alternative: Whether the test is one-sided (greater), one-sided
                   (lesser), or two-sided. If batch=True and 'f'
                   returns a scalar, or a single ndarray is supplied,
                   this argument is ignored as the test is always one-sided
                   in this configuration. Options:
                   - "greater"
                   - "lesser"
                   - "two-sided"
      permutation_type: The type of resampling procedure to use. Options are:
                        - "pairings": the order of observations is randomized,
                          but the sample that they belong to is preserved.
                        - "samples": the sample that an observation belongs to
                          is randomized, but the ordering is preserved.
                        - "independent": both the sample that an observation
                          belongs to and the ordering is randomized.
                        - "bootstrap": combines observations from all samples
                          and then resamples with replacement.
      less_is_more: True if smaller results should be considered more extreme
                    than larger results.
      iterations: Number of permutations to evaluate.
      batch: If set to True, 'f' accepts a list of ndarrays instead of a
             single array, and may return a scalar or a vector value.
      n_jobs: Number of workers to use. -1 indicates to use all available
              CPUs. If 'parallel' is not None, this parameter is ignored.
      parallel: An instance of joblib.Parallel.
      random_state: Either an integer >= 0 or an instance of
                    numpy.random.Generator. Used to attain reproducible
                    behavior.

    Returns:
      An instance of 'TestResult', where 'statistic' is the statistic
      calculated by 'f' on the input arrays, and 'p_value' is p-value
      calculated for the statistic.

    """

    args = [a] + list(args)
    if parallel is None:
        parallel = Parallel(n_jobs=n_jobs)

    if permutation_type == "independent":
        calc_permutation = partial(_ind_permutation, f, batch=batch)

    elif permutation_type == "samples":
        arg0_len = len(args[0])
        for arg in args:
            if len(arg) != arg0_len:
                raise ValueError("Samples must be the same size in a samples permutation test.")

        calc_permutation = partial(_samples_permutation, f, batch=batch)

    elif permutation_type == "pairings":
        calc_permutation = partial(_pairings_permutation, f, batch=batch)

    elif permutation_type == "bootstrap":
        calc_permutation = partial(_bootstrap_permutation, f, batch=batch)

    else:
        raise ValueError(f"Got unexpected value for permutation_type: {permutation_type}")

    if isinstance(random_state, (int, np.int64)):
        rng = np.random.default_rng(seed=random_state)

    elif isinstance(random_state, np.random.Generator):
        rng = random_state

    elif random_state is None:
        rng = np.random.default_rng()

    else:
        raise ValueError(f"Got unexpected value for random_state: {random_state}")

    seeds = rng.integers(0, MAX_INT, iterations)
    sample_statistic = delayed(calc_permutation)(args, shuffle=False)
    jobs = chain([sample_statistic], (delayed(calc_permutation)(args, seed) for seed in seeds))

    permutation_statistics = np.array(parallel(jobs))
    if permutation_statistics.ndim == 1:
        permutation_statistics = permutation_statistics.reshape(-1, 1)

    sample_statistic = permutation_statistics[0]
    permutation_statistics = permutation_statistics[1:]
    if len(sample_statistic) == 1:
        sample_statistic = sample_statistic[0]
        sample_delta = sample_statistic
        permutation_deltas = permutation_statistics

    elif len(args) > 2 or alternative == "two-sided":
        sample_delta = np.max(sample_statistic) - np.min(sample_statistic)
        permutation_deltas = np.max(permutation_statistics, axis=1) - np.min(
            permutation_statistics, axis=1
        )

    elif alternative == "greater":
        sample_delta = sample_statistic[0] - sample_statistic[1]
        permutation_deltas = permutation_statistics[:, 0] - permutation_statistics[:, 1]

    elif alternative == "lesser":
        sample_delta = sample_statistic[1] - sample_statistic[0]
        permutation_deltas = permutation_statistics[:, 1] - permutation_statistics[:, 0]

    else:
        raise ValueError(f"Got unexpected value for alternative: {alternative}")

    if less_is_more:
        pvalue = np.sum(sample_delta >= permutation_deltas) / iterations

    else:
        pvalue = np.sum(sample_delta <= permutation_deltas) / iterations

    return TestResult(statistic=sample_statistic, pvalue=pvalue)


def _permutation(permutate):
    @wraps(permutate)
    def new_func(calc_stat, args, seed=0, shuffle=True, batch=False):
        rng = np.random.default_rng(seed)
        if shuffle:
            args = permutate(args, rng)

        if batch:
            statistics = calc_stat(*args)

        else:
            statistics = tuple(calc_stat(x) for x in args)

        return statistics

    return new_func


# Evaluate a single permutation in an independent-style permutation test.
@_permutation
def _ind_permutation(args, rng):
    orig_shape = [x.shape for x in args]
    args = [arg.flatten() for arg in args]
    orig_size = [len(arg) for arg in args]
    args = np.concatenate(args)
    rng.shuffle(args)
    new_args = []
    for size, shape in zip(orig_size, orig_shape):
        new_arg = args[:size]
        new_arg.shape = shape
        args = args[size:]
        new_args.append(new_arg)

    return new_args


# Evaluate a single permutation in a pairings-style permutation test.
@_permutation
def _pairings_permutation(args, rng):
    shuffled_args = [rng.choice(x, size=len(x), replace=False) for x in args]
    return shuffled_args


# Evaluate a single permutation in a samples-style permutation test.
@_permutation
def _samples_permutation(args, rng):
    new_args = []
    for arg in zip(*args):
        arg = np.array(arg)
        rng.shuffle(arg)
        new_args.append(arg)

    return np.array(new_args).T


# Evaluate a single permutation in a bootstrap-style permutation test.
@_permutation
def _bootstrap_permutation(args, rng):
    orig_shape = [x.shape for x in args]
    args = [arg.flatten() for arg in args]
    orig_size = [len(arg) for arg in args]
    args = np.concatenate(args)
    new_args = []
    for size, shape in zip(orig_size, orig_shape):
        new_arg = rng.choice(args, size=size, replace=True)
        new_arg.shape = shape
        new_args.append(new_arg)

    return new_args


def confidence_interval(
    f: Callable[..., float],
    a: Union[np.ndarray, List[float]],
    *args: Union[np.ndarray, List[float]],
    iterations: int = 2000,
    levels: Tuple[float, ...] = (0.9, 0.95, 0.99),
    n_jobs: int = 1,
    parallel: Optional[Parallel] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[Dict[float, Tuple[float, float]], np.ndarray]:
    """
    Compute bootstrap confidence intervals for a statistic.

    Parameters:
        f (Callable): Function to compute the statistic of interest. It should accept
                      the groups `a` and `*args` as arguments.
        a (array-like): The first group of data.
        *args: Additional groups of data.
        iterations (int): Number of bootstrap iterations. Default is 2000.
                          Increase for higher precision, especially for extreme
                          confidence levels.
        levels (tuple): Confidence levels to compute (e.g., (0.9, 0.95, 0.99)).
        n_jobs (int): Number of parallel jobs. Default is 1 (no parallelization).
        parallel (joblib.Parallel, optional): Custom Parallel instance. If None, a new one is created.
        random_state (int, np.random.Generator, or None): Random seed or generator for reproducibility.

    Returns:
        tuple: A dictionary mapping confidence levels to confidence intervals,
               and the NumPy array of bootstrap statistics.

    Raises:
        ValueError: If `random_state` is of an unexpected type.

    """

    if parallel is None:
        parallel = Parallel(n_jobs=n_jobs)

    # Initialize random number generator
    if isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(seed=random_state)

    elif isinstance(random_state, np.random.Generator):
        rng = random_state

    elif random_state is None:
        rng = np.random.default_rng()

    else:
        raise ValueError(f"Got unexpected value for random_state: {random_state}")

    if iterations < 1:
        raise ValueError("Number of iterations must be at least 1.")

    if not levels:
        raise ValueError("At least one confidence level must be specified.")

    if not all(0 < level < 1 for level in levels):
        raise ValueError("All confidence levels must be between 0 and 1.")

    # Convert inputs to NumPy arrays
    groups = [np.array(x) for x in [a] + list(args)]

    if any(len(group) <= 1 for group in groups):
        raise ValueError(
            "All groups must contain at least two elements to compute meaningful confidence intervals."
        )

    # Warn if groups have unequal sizes
    if not np.all(np.array([len(g) for g in groups]) == len(groups[0])):
        msg = "Groups are not all equally sized! Ensure that this is appropriate for the statistic being computed."
        warnings.warn(msg, UserWarning)

    # Bootstrap resampling function
    def resample(seed: int) -> float:
        rng = np.random.default_rng(seed)
        resampled_groups = [rng.choice(g, size=len(g), replace=True) for g in groups]
        return f(*resampled_groups)

    # Generate bootstrap statistics
    max_int64 = np.iinfo(np.int64).max
    bootstrap_statistic = parallel(
        delayed(resample)(i) for i in rng.integers(max_int64, size=iterations)
    )

    # Calculate quantiles for confidence intervals
    quantiles = []
    for level in levels:
        quantiles.extend([(1 - level) / 2, 1 - (1 - level) / 2])

    intervals = list(batched(np.quantile(bootstrap_statistic, quantiles), 2))

    # Return confidence intervals and bootstrap statistics
    confidence_intervals = dict(zip(levels, intervals))
    return confidence_intervals, np.array(bootstrap_statistic)


def _compute_ci_width(
    model: Callable[[int, np.random.Generator], Any],
    calculate_statistic: Callable[..., float],
    level: float,
    kwargs: dict,
    n: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    x = model(n, rng)
    ci, _ = confidence_interval(
        calculate_statistic, *x, levels=(level,), random_state=rng, **kwargs
    )
    lower, upper = ci[level]
    return abs(upper - lower)


@singledispatch
def sample_size(
    model: Callable[[int, np.random.Generator], Sequence[Any]],
    calculate_statistic: Callable[..., float],
    width: float,
    level: float = 0.95,
    prob: float = 0.8,
    lower: int = 10,
    upper: int = 10000,
    mc_iterations: int = 2000,
    convergence_limit: int = 100,
    tol: float = 0.01,
    n_jobs: int = 1,
    parallel: Optional[Parallel] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    **kwargs: Any,
) -> int:
    """
    Estimate the required sample size for a target confidence interval width.

    Parameters:
        model (Callable): A function that simulates data, taking the sample size
                          (`n`) and random number generator (`rng`) as arguments,
                          and returning data to be passed to `calculate_statistic`.
                          Note that data should be returned as a list or tuple of
                          positional arguments to `calculate_statistic`.
        calculate_statistic (Callable): A function to compute the statistic of interest.
        width (float): Target width of the confidence interval.
        level (float): Confidence level for the interval. Must be between 0 and 1
                       (exclusive). Default is 0.95.
        prob (float): Quantile threshold for the Monte Carlo simulations. Must be
                      between 0 and 1 (inclusive). Default is 0.8.
        lower (int): Lower bound for the sample size. Must be >= 2. Default is 10.
        upper (int): Upper bound for the sample size. Must be > `lower`. Default is 10,000.
        mc_iterations (int): Number of Monte Carlo iterations per sample size. Must
                             be >= 1. Default is 2000.
        convergence_limit (int): Maximum iterations before stopping. Must be >= 0.
                                 Default is 100.
        tol (float): Tolerance for acceptable interval width. Must be >= 0. Default is 0.01.
        n_jobs (int): Number of parallel jobs. Default is 1 (no parallelization).
                      Use -1 for all available processors.
        parallel (joblib.Parallel, optional): Custom `Parallel` instance for
                                              parallelization. If None, a new instance
                                              is created with `n_jobs`.
        random_state (int, np.random.Generator, or None): Random seed or generator
                                                          for reproducibility. Default is None.
        **kwargs: Additional arguments passed to `confidence_interval`.

    Returns:
        int: The estimated sample size required to achieve the target confidence interval width.

    Raises:
        ValueError: If input parameters are invalid (e.g., out of range).
        RuntimeError: If the algorithm fails to converge within the given parameters.

    """

    logger = logging.getLogger(__name__)
    if lower < 2:
        raise ValueError(f"Argument for 'lower' must be >= 2. Got: {lower}")

    if upper <= lower:
        raise ValueError(f"Argument for 'upper' must be > 'lower'. Got: {upper}")

    if width < 0:
        raise ValueError(f"Argument for 'width' must be >= 0. Got: {width}")

    if not 0 <= prob <= 1:
        raise ValueError(f"Argument for 'prob' must be within range [0, 1]. Got: {prob}")

    if not 0 < level < 1:
        raise ValueError(f"Argument for 'level' must be within range (0, 1). Got: {level}")

    if mc_iterations < 1:
        raise ValueError(f"Argument for 'mc_iterations' must be >= 1. Got: {mc_iterations}")

    if convergence_limit < 0:
        raise ValueError(f"Argument for 'convergence_limit' must be >= 0. Got: {convergence_limit}")

    if tol < 0:
        raise ValueError(f"Argument for 'tol' must be >= 0. Got: {tol}")

    if tol * width < 1e-10:
        raise ValueError(f"The tolerance {tol} is too small for the target width {width}.")

    if n_jobs < -1:
        raise ValueError(f"Argument for 'n_jobs' must be >= -1. Got: {n_jobs}")

    if parallel is None:
        parallel = Parallel(n_jobs=n_jobs)

    # Initialize random number generator
    if isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(seed=random_state)

    elif isinstance(random_state, np.random.Generator):
        rng = random_state

    elif random_state is None:
        rng = np.random.default_rng()

    else:
        raise ValueError(f"Got unexpected value for random_state: {random_state}")

    max_int64 = np.iinfo(np.int64).max
    acceptable_range = (width - tol * width, width + tol * width)
    sample = partial(_compute_ci_width, model, calculate_statistic, level, kwargs)

    while upper - lower > 2 and convergence_limit > 0:
        convergence_limit -= 1
        n = (lower + upper) // 2
        seeds = rng.integers(max_int64, size=mc_iterations)
        widths = parallel(delayed(sample)(n, seed) for seed in seeds)
        upper_width = np.quantile(widths, prob)
        logger.debug("n: %d - upper_width: %f", n, upper_width)
        if acceptable_range[0] <= upper_width <= acceptable_range[1]:
            return n

        if upper_width <= width:
            upper = n

        else:
            lower = n

    raise RuntimeError(
        f"sample_size failed to converge for width={width}, level={level}, "
        f"tol={tol}, bounds=({lower}, {upper}), "
        f"current_n={n}, upper_width={upper_width}, acceptable_range={acceptable_range}."
    )


@sample_size.register
def _(
    data: np.ndarray,
    calculate_statistic: Callable[..., float],
    width: float,
    level: float = 0.95,
    prob: float = 0.8,
    lower: int = 10,
    upper: int = 10000,
    mc_iterations: int = 2000,
    convergence_limit: int = 100,
    tol: float = 0.01,
    n_jobs: int = 1,
    parallel: Optional[Parallel] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    **kwargs: Any,
) -> int:
    """
    Estimate the required sample size for a target confidence interval width using an empirical dataset.

    Parameters:
        data (np.ndarray): The empirical dataset to use for sampling.
        calculate_statistic (Callable): A function to compute the statistic of interest.
        width (float): Target width of the confidence interval.
        level (float): Confidence level for the interval. Must be between 0 and 1
                       (exclusive). Default is 0.95.
        prob (float): Quantile threshold for the Monte Carlo simulations. Must be
                      between 0 and 1 (inclusive). Default is 0.8.
        lower (int): Lower bound for the sample size. Must be >= 2. Default is 10.
        upper (int): Upper bound for the sample size. Must be > `lower`. Default is 10,000.
        mc_iterations (int): Number of Monte Carlo iterations per sample size. Must
                             be >= 1. Default is 2000.
        convergence_limit (int): Maximum iterations before stopping. Must be >= 0.
                                 Default is 100.
        tol (float): Tolerance for acceptable interval width. Must be >= 0. Default is 0.01.
        n_jobs (int): Number of parallel jobs. Default is 1 (no parallelization).
                      Use -1 for all available processors.
        parallel (joblib.Parallel, optional): Custom `Parallel` instance for
                                              parallelization. If None, a new instance
                                              is created with `n_jobs`.
        random_state (int, np.random.Generator, or None): Random seed or generator
                                                          for reproducibility. Default is None.
        **kwargs: Additional arguments passed to `confidence_interval`.

    Returns:
        int: The estimated sample size required to achieve the target confidence interval width.

    Raises:
        ValueError: If input parameters are invalid (e.g., out of range).
        RuntimeError: If the algorithm fails to converge within the given parameters.

    Notes:
        This implementation assumes the data provided is an empirical dataset from which
        bootstrap samples will be drawn. The `model` in this case is replaced with a
        function that samples data points with replacement from the empirical dataset.

    """

    def empirical_model(n, rng):
        return [rng.choice(data, size=n, replace=True)]

    return sample_size(
        empirical_model,
        calculate_statistic,
        width,
        level=level,
        prob=prob,
        lower=lower,
        upper=upper,
        mc_iterations=mc_iterations,
        convergence_limit=convergence_limit,
        tol=tol,
        n_jobs=n_jobs,
        parallel=parallel,
        random_state=random_state,
        **kwargs,
    )


def tail_weight(x, side="both"):
    """
    Estimate the tail weight of a dataset using the L/RMC method.

    Args:
      x: A 1-dimensional ndarray with dtype 'float64'.
      side: Whether to calculate the tail weight for the left or right
            side of the distribution or both. Options:
            - "left"
            - "right"
            - "both"

    Return:
      A floating-point number between -1 and 1 indicating whether the
      data is tail light (-1), normal (0), or tail heavy (+1).

    """

    x = np.sort(x, kind="mergesort")
    midpoint = int(len(x) / 2)
    if side == "left":
        x_left = x[:midpoint]
        lmc = medcouple(x_left) * -1
        return lmc

    if len(x) % 2 == 0:
        x_right = x[midpoint:]

    else:
        x_right = x[midpoint + 1 :]

    rmc = medcouple(x_right)
    if side == "right":
        return rmc

    if side == "both":
        x_left = x[:midpoint]
        lmc = medcouple(x_left) * -1
        return (lmc + rmc) / 2

    raise ValueError(f"Unrecognized value for parameter `side`: {side}")


def standard_error(f, x, iterations=1000, random_state=None):
    """
    Calculate the standard error of a statistic f of a sample x.

    Args:
      f: A function that takes a single ndarray-like object and returns
         a scalar value representing a sample statistic.
      x: A ndarray of data to use in calculating the standard error of f.
      iterations: Number of bootstrap resamples to perform. Default is 1000.
      random_state: Either an integer >= 0 or an instance of
                    numpy.random.Generator. Used to attain reproducible
                    behavior.

    Returns:
      A float representing the standard error of f of x.

    """

    x = np.array(x)
    if isinstance(random_state, int):
        rng = np.random.default_rng(seed=random_state)

    elif isinstance(random_state, np.random.Generator):
        rng = random_state

    elif random_state is None:
        rng = np.random.default_rng()

    else:
        raise ValueError(f"Got unexpected value for random_state: {random_state}")

    bootstrap_statistics = [f(rng.choice(x, size=len(x))) for _ in range(iterations)]
    return np.std(bootstrap_statistics, ddof=1)


def gcv(x: ArrayLike) -> float:
    """
    Calculate the geometric coefficient of variation for a sample x.

    Args:
      x: An ndarray array of data to calculate the gcv of.

    Returns:
      The gcv of x.

    """

    return math.sqrt(math.expm1(np.var(np.log(x))))


def hvar(x: ArrayLike) -> float:
    """
    Calculate the harmonic variability for a sample x.

    Args:
      x: An ndarray of data to calculate the harmonic variability of.

    Returns:
      The harmonic variability of x.

    """

    x = np.array(x)
    x_inv = 1 / x
    return (np.var(x_inv) / np.mean(x_inv) ** 4) / len(x)


def xtrim(x, trim_amount=0.1):
    """
    Trim an equal amount of data from the left and right sides of x.

    Args:
      x: An ndarray of values to trim data from.
      trim_amount: The amount of data to trim from x before calculating
                   the statistic. Default is 0.1.

    Returns:
      The trimmed dataset x.

    """

    x = np.array(x)
    lower_quantile = trim_amount / 2
    quantiles = lower_quantile, 1 - lower_quantile
    lower_threshold, upper_threshold = np.quantile(x, quantiles)
    trimmed_x = x[np.logical_and(x > lower_threshold, x < upper_threshold)]

    return trimmed_x


def ftrim(f, x, trim_amount=0.1):
    """
    Calculate the trimmed statistic f of dataset x.

    Args:
      f: A function that takes a single ndarray-like object and returns
         a scalar value representing a sample statistic.
      x: A ndarray of data to calculate the trimmed statistic of.
      trim_amount: The amount of data to trim from x before calculating
                   the statistic. Default is 0.1.

    Returns:
      The trimmed statistic f of x.

    """

    trimmed_x = xtrim(x, trim_amount=trim_amount)
    return f(trimmed_x)


def sum_prob(p: Sequence[float]) -> float:
    """
    Apply the sum rule of probability to an array of independent
    probabilities.

    """

    if len(p) == 0:
        return 0.0

    return reduce(lambda x, y: x + y - x * y, p)


def mutual_info(
    x: ArrayLike, y: ArrayLike, base: Union[int, float] = 2, norm: str = "min"
) -> float:
    """
    Calculate the mutual information between two discrete variables.

    Mutual information quantifies the amount of information obtained about one random variable by
    observing the other random variable. The function can also normalize the mutual information with respect to the
    minimum, maximum, or average of the entropies of the variables.

    Args:
      x: A list or NumPy array of samples from the first random variable.
      y: A list or NumPy array of samples from the second random variable. Must be the same length as x.
      base: The logarithmic base to use when calculating the mutual information. Default is 2 (binary log).
      norm: A string indicating the type of normalization to apply to the mutual information. Options:
            - "min": Normalize by the minimum entropy of the two variables (Default).
            - "max": Normalize by the maximum entropy of the two variables.
            - "avg": Normalize by the average entropy of the two variables.
            - "none": No normalization.

    Returns:
      float: The calculated mutual information. If normalization is applied, the mutual information
             is divided by the specified entropy measure.

    Raises:
      ValueError: If the input arrays x and y are not of the same length.

    """

    if base <= 0:
        raise ValueError("'base' must be a number greater than 0.")

    if len(x) != len(y):
        raise ValueError("Arrays 'x' and 'y' must be the same length.")

    if norm not in ("min", "max", "avg", "none"):
        raise ValueError(
            f"Argument for 'norm' must be one of 'min', 'max', 'avg', or 'none'. Got: {norm}"
        )

    x = np.array(x)
    y = np.array(y)
    mutual_information = 0
    x_labels = np.unique(x)
    y_labels = np.unique(y)
    px = [np.mean(x == xi) for xi in x_labels]
    py = [np.mean(y == yi) for yi in y_labels]
    for pxi, xi in zip(px, x_labels):
        for pyi, yi in zip(py, y_labels):
            pxy = np.mean(np.logical_and(x == xi, y == yi))
            if pxy != 0:
                mutual_information += pxy * (np.log2(pxy / (pxi * pyi)) / np.log2(base))

    if norm != "none":
        x_entropy = stats.entropy(px, base=base)
        y_entropy = stats.entropy(py, base=base)
        if x_entropy == 0 and y_entropy == 0:
            return 0.0

        elif norm == "avg":
            normalization_factor = (x_entropy + y_entropy) / 2

        elif x_entropy == 0 or y_entropy == 0:
            return 0.0

        elif norm == "min":
            normalization_factor = min(x_entropy, y_entropy)

        else:
            normalization_factor = max(x_entropy, y_entropy)

        mutual_information = mutual_information / normalization_factor

    return mutual_information
