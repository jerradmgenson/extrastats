"""
extrastats is a library of high-quality statistical routines that are
missing from the mainstream Python packages.

Copyright 2022-2023 Jerrad Michael Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import logging
import math
import numbers
import warnings
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from itertools import batched, chain
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from kneed import KneeLocator
from numpy.typing import ArrayLike, NDArray

try:
    from robustats import medcouple

except ImportError:
    warnings.warn(
        "Can't import robustats. Using 'medcouple' implementation from statsmodels.", UserWarning
    )

    from statsmodels.stats.stattools import medcouple

from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.tree import DecisionTreeRegressor

DEFAULT_THRESHOLD = 1.5
MAX_INT = 2147483648

DistSide = Enum("DistSide", "left right both")
Alternative = Enum("Alternative", "greater less two_sided")
PermutationType = Enum("PermutationType", "independent samples pairings bootstrap")

TestResult = namedtuple("TestResult", "statistic pvalue")


class MedcoupleError(Exception):
    """
    Raised when there is an error in the medcouple calculation.

    """


@singledispatch
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


@adjusted_boxplot.register
def _(
    df: pd.DataFrame,
    k=DEFAULT_THRESHOLD,
    frac=1,
    n_jobs=1,
    parallel=None,
    random_state=None,
):
    """
    An adjusted_boxplot variant that takes a pandas dataframe and return
    a nested list of boolean values indicating which values in each
    column are outliers.

    Args:
      df: A pandas dataframe.
      k: Factor for calculating outlier thresholds.
         Default value is 1.5.
      frac: Fraction of the data to use for calculating the medcouple.
            When set to 1, the entire array is used.
      sparse: Currently not implemented for this variant of
              'adjusted_boxplot'.
      n_jobs: Number of workers to use. -1 indicates to use all available
              CPUs. If 'parallel' is not None, this parameter is ignored.
      parallel: An instance of joblib.Parallel.
      random_state: Either an integer >= 0 or an instance of
                    numpy.random.Generator. Used to attain reproducible
                    behavior when frac < 1.

    Returns:
      A tuple of (low, high) outlier thresholds. If 'k' is a sequence,
      a generator of tuples is returned instead.

    """

    if parallel is None:
        parallel = Parallel(n_jobs=n_jobs)

    aboxplt = partial(adjusted_boxplot, k=k, frac=frac, random_state=random_state)

    jobs = (df[col].to_numpy(dtype=float) for col in df)
    outliers = parallel(delayed(aboxplt)(job) for job in jobs)
    if isinstance(k, numbers.Number):
        return pd.DataFrame(dict(zip(df.columns, outliers)))

    dfs = dict()
    for i, t in enumerate(k):
        data = (o[i] for o in outliers)
        dfs[t] = pd.DataFrame(dict(zip(df.columns, data)))

    return dfs


def permutation_test(
    f,
    a,
    *args,
    alternative=Alternative.two_sided,
    permutation_type=PermutationType.bootstrap,
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
                   (lesser), or two-sided. Should be a member of
                   'Alternative'. If batch=True and 'f' returns a scalar,
                   or a single ndarray is supplied,  this argument is
                   ignored as the test is always one-sided in this
                   configuration.
      permutation_type: A member of 'PermutationType'. The various
                        options are:
                        - pairings: the order of observations is randomized,
                          but the sample that they belong to is preserved.
                        - samples: the sample that an observation belongs to
                          is randomized, but the ordering is preserved.
                        - independent: both the sample that an observation
                          belongs to and the ordering is randomized.
                        - bootstrap: combines observations from all samples
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

    if permutation_type == PermutationType.independent:
        calc_permutation = partial(_ind_permutation, f, batch=batch)

    elif permutation_type == PermutationType.samples:
        arg0_len = len(args[0])
        for arg in args:
            if len(arg) != arg0_len:
                raise ValueError("Samples must be the same size in a samples permutation test.")

        calc_permutation = partial(_samples_permutation, f, batch=batch)

    elif permutation_type == PermutationType.pairings:
        calc_permutation = partial(_pairings_permutation, f, batch=batch)

    elif permutation_type == PermutationType.bootstrap:
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

    elif len(args) > 2 or alternative == Alternative.two_sided:
        sample_delta = np.max(sample_statistic) - np.min(sample_statistic)
        permutation_deltas = np.max(permutation_statistics, axis=1) - np.min(
            permutation_statistics, axis=1
        )

    elif alternative == Alternative.greater:
        sample_delta = sample_statistic[0] - sample_statistic[1]
        permutation_deltas = permutation_statistics[:, 0] - permutation_statistics[:, 1]

    elif alternative == Alternative.less:
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

        if hasattr(calc_stat, "_accepts_random_state"):
            calc_stat = partial(calc_stat, random_state=rng)

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


def accepts_random_state(f):
    """
    Used as a decorator, this function indicates to 'permutation_test'
    that the given function, 'f', has a 'random_state' parameter.
    The function must be able to accept an instance of
    numpy.random.Generator as an argument to random_state.

    """

    f._accepts_random_state = True
    return f


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


def iqr(x):
    """
    Calculate the interquartile range of the given ndarray.

    x: The ndarray to calculate the IQR of. If the array is not 1D, it
       will be flattened prior to calculating the IQR.

    Returns:
      The IQR of x.

    """

    if x.ndim > 1:
        x = x.flatten()

    x = np.sort(x, kind="mergesort")
    midpoint = int(len(x) / 2)
    x_lower = x[:midpoint]
    if len(x) % 2 == 0:
        x_upper = x[midpoint:]

    else:
        x_upper = x[midpoint + 1 :]

    q1 = np.median(x_lower)
    q3 = np.median(x_upper)
    return q3 - q1


def tail_weight(x, side=DistSide.both):
    """
    Estimate the tail weight of a dataset using the L/RMC method.

    Args:
      x: A 1-dimensional ndarray with dtype 'float64'.
      side: Whether to calculate the tail weight for the left or right
            side of the distribution or both. Should be a member of
            'DistSide'.

    Return:
      A floating-point number between -1 and 1 indicating whether the
      data is tail light (-1), normal (0), or tail heavy (+1).

    """

    x = np.sort(x, kind="mergesort")
    midpoint = int(len(x) / 2)
    if side == DistSide.left:
        x_left = x[:midpoint]
        lmc = medcouple(x_left) * -1
        return lmc

    if len(x) % 2 == 0:
        x_right = x[midpoint:]

    else:
        x_right = x[midpoint + 1 :]

    rmc = medcouple(x_right)
    if side == DistSide.right:
        return rmc

    if side == DistSide.both:
        x_left = x[:midpoint]
        lmc = medcouple(x_left) * -1
        return (lmc + rmc) / 2

    raise ValueError(f"Unrecognized value for parameter `side`: {side}")


# Calculate mutual information on discrete datasets.
def _mutual_info_discrete(a, b, average_method="arithmetic"):
    return adjusted_mutual_info_score(a, b, average_method=average_method)


# Calculate mutual information on continuous datasets.
@accepts_random_state
def _mutual_info_continuous(a, b, discrete_features=False, n_neighbors=3, random_state=None):
    a = a.reshape(-1, 1)
    return mutual_info_regression(
        a,
        b,
        discrete_features=discrete_features,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )


def test_mutual_info(
    a,
    b,
    a_discrete=True,
    b_discrete=True,
    average_method="arithmetic",
    n_neighbors=3,
    random_state=None,
    **kwargs,
):
    """
    Conduct a permutation test of mutual information between groups a and b.

    Args:
      a: An ndarray of data for group a.
      b: An ndarray of data for group b.
      a_discrete: Indicates that group a represents a discrete variable.
                  Default is True.
      b_discrete: Indicates that group b represents a discrete variable.
                  Default is True.
      average_method: The averaging method to use when at least one of the
                      groups represents a continuous variable.
                      Default is 'arithmetic'.
      n_neighbors: Value for k when calculating k-nearest neighbors distances
                   for continuous variables. Default is 3.
      random_state: Either an integer >= 0 or an instance of
                    numpy.random.Generator. Used to attain reproducible
                    behavior. Default is None.
      **kwargs: Any additional keyword arguments are passed along to
                permutation_test.

    Returns:
      An instance of TestResult. See permutation_test docstring for details.

    """

    rs = random_state
    if isinstance(random_state, np.random.Generator):
        rs = random_state.integers(0, MAX_INT)

    if a_discrete and b_discrete:
        f = partial(_mutual_info_discrete, average_method=average_method)

    elif a_discrete:
        f = partial(
            _mutual_info_continuous,
            discrete_features=True,
            n_neighbors=n_neighbors,
            random_state=rs,
        )

    elif b_discrete:
        a, b = b, a
        f = partial(
            _mutual_info_continuous,
            discrete_features=True,
            n_neighbors=n_neighbors,
            random_state=rs,
        )

    else:
        f = partial(
            _mutual_info_continuous,
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=rs,
        )

    return permutation_test(
        f,
        a,
        b,
        batch=True,
        permutation_type=PermutationType.pairings,
        random_state=random_state,
        **kwargs,
    )


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


OptrimResult = namedtuple("OptrimResult", "statistic standard_error trim_amount")


def optrim(f, x, max_trim_amount=0.25, sensitivity=1, se_iterations=1000, random_state=None):
    """
    Calculate a trimmed statistic f of dataset x, using the standard error
    and the kneedle method to find the optimal trim amount.

    Args:
      f: A function that takes a single ndarray-like object and returns
         a scalar value representing a sample statistic.
      x: A ndarray of data to calculate the trimmed statistic of.
      max_trim_amount: The maximum amount of data that can be trimmed
                       from x. Default is 0.25.
      sensitivity: The S parameter of the kneedle algorithm. Default is 1.
      se_iterations: Number of resamples to use for calculating the
                     standard error. Default is 1000.
      random_state: Either an integer >= 0 or an instance of
                    numpy.random.Generator. Used to attain reproducible
                    behavior.

    Returns:
      An OptrimResult object, which contains the trimmed statistic of x,
      the standard error of the statistic, and the trim amount.

    """

    x = np.array(x)
    results = []
    for trim_amount in range(int(max_trim_amount * 100) + 1):
        trim_frac = trim_amount / 100
        if trim_amount == 0:
            trimmed_x = x

        else:
            trimmed_x = xtrim(x, trim_amount=trim_frac)

        results.append(
            OptrimResult(
                statistic=f(trimmed_x),
                standard_error=standard_error(
                    f, trimmed_x, iterations=se_iterations, random_state=random_state
                ),
                trim_amount=trim_frac,
            )
        )

    x = [r.trim_amount for r in results]
    y = [r.standard_error for r in results]
    knee_locator = KneeLocator(
        x, y, curve="convex", direction="decreasing", online=True, S=sensitivity
    )

    return results[int(knee_locator.knee * 100)]


def sum_prob(p: Sequence[float]) -> float:
    """
    Apply the sum rule of probability to an array of independent
    probabilities.

    """

    if len(p) == 0:
        return 0.0

    return reduce(lambda x, y: x + y - x * y, p)


@dataclass
class BinnedData:
    data: NDArray[int]
    edges: NDArray[float]
    error: float


def tree_bin(
    x: ArrayLike, min_bins: int = 2, max_bins: int = 40, random_state: int = 0
) -> BinnedData:
    """
    Bins the given array 'x' by training a decision tree regressor and using its leaves as bin edges.

    The function first fits a full decision tree regressor to the data and then applies cost complexity
    pruning to find the optimal tree size within the specified range of bins. It uses the tree's leaves
    to define bin edges and then assigns each element in 'x' to a bin. It also computes the root mean
    square error of the binning as an error metric.

    Args:
      x (NDArray[float]): The input array to bin, should be a 1D array of floats.
      min_bins (int, optional): The minimum number of bins to consider. Defaults to 2.
      max_bins (int, optional): The maximum number of bins to consider. Defaults to 40.
      random_state (int, optional): A seed for the random number generator for reproducibility. Defaults to 0.

    Returns:
      BinnedData: A dataclass instance containing the following fields:
        - data (NDArray[int]): An array of the same shape as 'x', where each element is the index of the bin it belongs to.
        - edges (NDArray[float]): The edges of the bins as determined by the decision tree.
        - error (float): The root mean square error of the binning.

    Raises:
      ValueError: If the input array 'x' is empty, indicating that binning cannot be performed.

    """

    x = np.array(x)
    if x.size == 0:
        msg = "Input array 'x' is empty. Binning cannot be performed on an empty array."
        raise ValueError(msg)

    unique_values = np.unique(x)
    if unique_values.size < 5 or x.size < 10:
        warning_msg = (
            f"Input array 'x' has less than 5 unique values or is smaller than 10 elements "
            f"({'size' if x.size < 10 else 'unique count'} is {min(x.size, unique_values.size)}). "
            "Using simple integer encoding instead."
        )

        warnings.warn(warning_msg, UserWarning)
        labels, binned_data = np.unique(x, return_inverse=True)
        return BinnedData(data=binned_data, edges=labels, error=0.0)

    if min_bins < 2:
        raise ValueError("'min_bins' can not be less than 2.")

    if max_bins <= min_bins:
        raise ValueError("'max_bins' must be greater than 'min_bins'.")

    x2d = x.reshape(-1, 1)

    # Fit the full tree and get the effective alphas
    full_tree = DecisionTreeRegressor(random_state=random_state)
    full_tree.fit(x2d, x)
    path = full_tree.cost_complexity_pruning_path(x2d, x)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # Prepare dicts for the number of leaves and MSE scores
    leaves_scores = {}
    trees = {}

    # Iterate over the effective alphas and collect the number of leaves and MSE scores
    for ccp_alpha in reversed(ccp_alphas):  # Reverse to start with the smallest tree
        tree = DecisionTreeRegressor(random_state=random_state, ccp_alpha=ccp_alpha)
        tree.fit(x2d, x)
        mse = np.mean((tree.predict(x2d) - x) ** 2)
        leaves = tree.get_n_leaves()
        if (
            min_bins <= leaves <= max_bins
        ):  # Check if the number of leaves is within the desired range
            leaves_scores[leaves] = mse
            trees[leaves] = tree

        elif leaves > max_bins:
            break

    # num_leaves now contains the number of leaves for trees in the range of 2 to 40 leaves
    # mse_scores contains the corresponding MSE scores
    # You can now plot or analyze these to choose the best tree size
    try:
        selected_num_leaves = round(
            _locate_elbow(list(leaves_scores.values()), list(leaves_scores))
        )
        if selected_num_leaves == max_bins:
            msg = "kneedle method doesn't appear to have converged. Using fallback method."
            raise RuntimeError(msg)

    except RuntimeError as rte:
        warnings.warn(str(rte), RuntimeWarning)
        selected_num_leaves = max(min(round(np.log(len(x))), max_bins), min_bins)

    selected_tree = trees[selected_num_leaves]
    edges = _get_leaf_edges(selected_tree)
    left_edges = np.unique([edge[0] for edge in edges if np.isfinite(edge[0])])
    binned_data = np.digitize(x, left_edges)

    return BinnedData(binned_data, left_edges, np.sqrt(leaves_scores[selected_num_leaves]))


def _get_leaf_edges(
    tree: DecisionTreeRegressor,
    node_id: int = 0,
    left_edge: float = -np.inf,
    right_edge: float = np.inf,
) -> List[Tuple[float, float]]:
    """
    Recursively traverses a decision tree to extract the edge values of each leaf node.

    The function traverses the decision tree starting from the given node_id and collects
    the left and right edge values for each leaf node encountered during the traversal. The
    edge values are inclusive for the left edge and exclusive for the right edge, defining
    the intervals represented by the leaf nodes of the tree.

    Args:
      tree (DecisionTreeRegressor): The trained decision tree regressor from which to extract leaf edges.
      node_id (int, optional): The ID of the current node being traversed. Defaults to the root node with ID 0.
      left_edge (float, optional): The left edge value of the current node interval. Defaults to -infinity, representing the leftmost edge of the tree.
      right_edge (float, optional): The right edge value of the current node interval. Defaults to infinity, representing the rightmost edge of the tree.

    Returns:
      List[Tuple[float, float]]: A list of tuples, where each tuple contains the (left_edge, right_edge) values corresponding to a leaf node.

    Note:
      This function is meant to be used internally within the decision tree binning process and relies on the internal structure of the
      sklearn.tree.DecisionTreeRegressor, which is not part of the public API and could change in future releases of scikit-learn.
    """

    # Check if we have a leaf
    if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id] == -1:
        # Return the edges corresponding to the leaf
        return [(left_edge, right_edge)]
    else:
        edges = []
        # Left child
        left_child = tree.tree_.children_left[node_id]
        left_threshold = tree.tree_.threshold[node_id]
        edges.extend(_get_leaf_edges(tree, left_child, left_edge, left_threshold))

        # Right child
        right_child = tree.tree_.children_right[node_id]
        right_threshold = tree.tree_.threshold[node_id]
        edges.extend(_get_leaf_edges(tree, right_child, right_threshold, right_edge))

        return edges


def _locate_elbow(y: ArrayLike, x: Optional[ArrayLike] = None) -> Union[int, float]:
    """
    Attempts to find the elbow point in a dataset using the kneedle method, which is indicative of the
    'knee' or 'elbow' in a convex decreasing curve.

    The function first checks if the data exhibits a generally decreasing and convex pattern. If so,
    it applies the kneedle algorithm to locate the elbow point. The kneedle algorithm is useful for
    determining the point of maximum curvature in such a curve, often used in the context of determining
    optimal binning, cluster number, etc.

    Args:
      y (list or np.ndarray): The y-values of the curve, expected to be in a convex decreasing form.
      x (list or np.ndarray, optional): The x-values corresponding to the y-values. If not provided, the
                                        indices of the y-values are used.

    Returns:
      int or float: The x-value corresponding to the elbow point in the curve.

    Raises:
      RuntimeError: If the kneedle method fails to identify an elbow point due to the data not being
                    decreasing or convex as required.

    Note:
    This function relies on the 'KneeLocator' from the 'kneed' package and 'kendalltau' from 'scipy.stats'.
    It should be noted that the thresholds for determining if the data is decreasing and convex have been
    empirically determined and may not be universally applicable.

    Example:
    >>> y = [0, 2, 4, 6, 7, 8, 9, 9, 10]
    >>> _locate_elbow(y)
    4  # This would return the index of '7' in the list, as the hypothetical elbow point.

    """

    # These values were determined empirically via Monte Carlo simulation.
    decreasing_threshold = -0.2
    convex_threshold = 0.1

    # First check to see if this data can be used with the kneedle method.
    decreasing = stats.kendalltau(np.arange(len(y)), y)[0] <= decreasing_threshold
    y_deriv = np.gradient(y, np.arange(len(y)))
    convex = stats.kendalltau(np.arange(len(y_deriv)), y_deriv)[0] >= convex_threshold
    if decreasing and convex:
        if x is None:
            x = np.arange(len(y))

        kneedle = KneeLocator(x, y, curve="convex", direction="decreasing")
        if kneedle.elbow is not None:
            return kneedle.elbow

    msg = f"kneedle method failed to locate optimal bins. Decreasing: {decreasing} Convex: {convex}"
    raise RuntimeError(msg)


def plot_bins(data: ArrayLike, bin_edges: ArrayLike) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the distribution of the data and overlay the bin edges.

    This function creates a histogram to visualize the distribution of the data
    and adds vertical lines representing the bin edges. The plot is displayed
    on screen, and the function returns the matplotlib figure and axes objects
    for further customization or saving.

    Args:
      data: np.ndarray
        The 1D array of data points to be binned and plotted.
      bin_edges: np.ndarray
        An array of bin edge values to be plotted as vertical lines on the histogram.

    Returns:
      Tuple[plt.Figure, plt.Axes]
        A tuple containing the figure and axes objects of the plot.

    """

    data = np.array(data)
    bin_edges = np.array(bin_edges)

    # Create histogram of the data
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, alpha=0.5, color="blue", label="Data histogram")

    # Add vertical lines for the bin edges
    for edge in bin_edges:
        ax.axvline(edge, color="red", linestyle="dashed", linewidth=2, label="Bin edge")

    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram with Bin Edges")

    # We use a trick to create a single legend entry for bin edges
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.show()

    return fig, ax


MINorm = Enum("MINorm", ("min_info", "max_info", "avg_info", "none"))


def mutual_info(
    x: ArrayLike, y: ArrayLike, base: Union[int, float] = 2, norm: MINorm = MINorm.min_info
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
      norm: An enum indicating the type of normalization to apply to the mutual information. Options are:
            - MINorm.min_info: Normalize by the minimum entropy of the two variables (Default).
            - MINorm.max_info: Normalize by the maximum entropy of the two variables.
            - MINorm.avg_info: Normalize by the average entropy of the two variables.
            - MINorm.none: No normalization.

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

    x = np.array(x)
    y = np.array(y)
    norm = MINorm(norm)

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

    if norm != MINorm.none:
        x_entropy = stats.entropy(px, base=base)
        y_entropy = stats.entropy(py, base=base)
        if x_entropy == 0 and y_entropy == 0:
            return 0.0

        elif norm == MINorm.avg_info:
            normalization_factor = (x_entropy + y_entropy) / 2

        elif x_entropy == 0 or y_entropy == 0:
            return 0.0

        elif norm == MINorm.min_info:
            normalization_factor = min(x_entropy, y_entropy)

        else:
            normalization_factor = max(x_entropy, y_entropy)

        mutual_information = mutual_information / normalization_factor

    return mutual_information
