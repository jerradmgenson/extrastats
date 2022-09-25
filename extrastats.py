"""
extrastats implements all the statistics you've been missing from numpy,
scipy, and statsmodels! Well, maybe not all of them. But some of them.

Copyright 2022 Jerrad Michael Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import math
import numbers
from enum import Enum
from collections import namedtuple
from typing import Sequence
from functools import partial, singledispatch, wraps
from itertools import chain

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from joblib import Parallel, delayed
from robustats import medcouple
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from kneed import KneeLocator

DEFAULT_THRESHOLD = 1.5
MAX_INT = 2147483648

DistSide = Enum("DistSide", "left right both")
Alternative = Enum("Alternative", "greater less two_sided")
PermutationType = Enum("PermutationType", "independent samples pairings bootstrap")

TestResult = namedtuple("TestResult", "statistic pvalue")


@singledispatch
def adjusted_boxplot(
    x,
    k=DEFAULT_THRESHOLD,
    frac=1,
    n_jobs=1,
    parallel=None,
    random_state=None,
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

    def apply_threshold(k):
        if mc >= 0:
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
                raise ValueError(
                    "Samples must be the same size in a samples permutation test."
                )

        calc_permutation = partial(_samples_permutation, f, batch=batch)

    elif permutation_type == PermutationType.pairings:
        calc_permutation = partial(_pairings_permutation, f, batch=batch)

    elif permutation_type == PermutationType.bootstrap:
        calc_permutation = partial(_bootstrap_permutation, f, batch=batch)

    else:
        raise ValueError(
            f"Got unexpected value for permutation_type: {permutation_type}"
        )

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
    jobs = chain(
        [sample_statistic], (delayed(calc_permutation)(args, seed) for seed in seeds)
    )

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
def _mutual_info_continuous(
    a, b, discrete_features=False, n_neighbors=3, random_state=None
):
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


def gcv(x):
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


def optrim(
    f, x, max_trim_amount=0.25, sensitivity=1, se_iterations=1000, random_state=None
):
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

    accum = p[0]
    p = p[1:]
    while len(p):
        accum = accum + p[0] - accum * p[0]
        p = p[1:]

    return accum
