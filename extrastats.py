import math
from enum import Enum
from collections import namedtuple
from functools import reduce, partial, singledispatch
from itertools import chain
import numbers

import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from robustats import medcouple
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.feature_selection import mutual_info_regression

DEFAULT_THRESHOLD = 1.5
INT_MAX = 2147483648

DistSide = Enum('DistSide', 'left right both')
Alternative = Enum('Alternative', 'greater lesser two_sided')
PermutationType = Enum('PermutationType', 'independent samples pairings')

TestResult = namedtuple('TestResult', 'statistic p_value')


@singledispatch
def adjusted_boxplot(x,
                     threshold=DEFAULT_THRESHOLD,
                     frac=1,
                     sparse=False,
                     n_jobs=1,
                     parallel=None,
                     random_state=None):
    """
    Apply the adjusted boxplot method on an array of numeric values.

    Args:
      x: A 1-D ndarray of numeric values.
      threshold: Specifies the numeric "fence" for detecting outliers.
                 Default is 1.5. Can also be a sequence of thresholds.
      frac: Fraction of the data to use for calculating the medcouple.
            When set to 1, the entire array is used.
      sparse: When set to True, scipy.sparse.coo_arrays are returned
              instead of ndarrays.
      n_jobs: Not used in this variant of adjusted_boxplot.
      parallel: Not used in this variant of adjusted_boxplot.
      random_state: Either an integer >= 0 or an instance of
                    numpy.random.Generator. Used to attain reproducible
                    behavior when frac < 1.

    Returns:
      A boolean array the same shape as 'x' where True elements indicate
      outliers. If 'threshold' is a sequence, a generator of arrays is
      returned instead.

    """

    if x.ndim > 2:
        raise ValueError('adjusted_boxplot called with x.ndim > 2')

    x0 = x[~np.isnan(x)]
    if isinstance(random_state, int):
        rng = np.random.default_rng(seed=random_state)

    elif isinstance(random_state, np.random.Generator):
        rng = random_state

    elif random_state is None:
        rng = np.random.default_rng()

    else:
        raise ValueError(f'Got unexpected value for random_state: {random_state}')

    if frac < 1:
        x0 = rng.choice(x0, size=int(round(frac*len(x))), replace=False)

    q1, q3 = np.quantile(x0, [.25, .75])
    iqr = q3 - q1
    mc = medcouple(x0)

    def apply_threshold(threshold):
        if mc >= 0:
            lower_fence = q1 - threshold * np.exp(-3.5 * mc) * iqr
            upper_fence = q3 + threshold * np.exp(4 * mc) * iqr

        else:
            lower_fence = q1 - threshold * np.exp(-4 * mc) * iqr
            upper_fence = q3 + threshold * np.exp(3.5 * mc) * iqr

        outliers = (x < lower_fence) | (x > upper_fence)
        assert outliers.shape == x.shape
        if sparse:
            return sp.sparse.coo_array(outliers)

        return outliers

    try:
        outlier_groups = (apply_threshold(t) for t in threshold)
        return outlier_groups

    except TypeError:
        return apply_threshold(threshold)


@adjusted_boxplot.register
def _(df: pd.DataFrame,
      threshold=DEFAULT_THRESHOLD,
      frac=1,
      sparse=False,
      n_jobs=1,
      parallel=None,
      random_state=None):
    """
    An adjusted_boxplot variant that takes a pandas dataframe and return
    a nested list of boolean values indicating which values in each
    column are outliers.

    Args:
      df: A pandas dataframe.
      threshold: Specifies the numeric "fence" for detecting outliers.
                 Default is 1.5. Can also be a sequence of thresholds.
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
      A Pandas dataframe with the same length and columns as 'df' where
      True elements indicate outliers. If 'threshold' is a sequence, a
      generator of dataframes is returned instead.

    """

    if parallel is None:
        parallel = Parallel(n_jobs=n_jobs)

    aboxplt = partial(adjusted_boxplot,
                      threshold=threshold,
                      frac=frac,
                      random_state=random_state)

    jobs = (df[col].to_numpy(dtype=float) for col in df)
    outliers = parallel(delayed(aboxplt)(job) for job in jobs)
    if isinstance(threshold, numbers.Number):
        return pd.DataFrame({k: v for k, v in zip(df.columns, outliers)})

    else:
        dfs = dict()
        for i, t in enumerate(threshold):
            data = (o[i] for o in outliers)
            dfs[t] = pd.DataFrame({k: v for k, v in zip(df.columns, data)})

        return dfs


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

    if side == DistSide.left:
        x_left = x[x < np.median(x)]
        mc = medcouple(x_left)
        return mc * -1

    elif side == DistSide.right:
        x_right = x[x > np.median(x)]
        mc = medcouple(x_right)
        return mc

    elif side == DistSide.both:
        median = np.median(x)
        x_left = x[x < median]
        lmc = medcouple(x_left) * -1
        x_right = x[x > median]
        rmc = medcouple(x_right)
        return (lmc + rmc) / 2

    else:
        raise ValueError(f'Unrecognized value for parameter `side`: {side}')


def permutation_test(f, a, *args,
                     alternative=Alternative.two_sided,
                     permutation_type=PermutationType.independent,
                     iterations=1000,
                     batch=False,
                     n_jobs=1,
                     parallel=None,
                     random_state=None):
    """
    Conduct a randomized permutation test on the given datasets.

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
                raise ValueError('Samples must be the same size in a samples permutation test.')

        calc_permutation = partial(_samples_permutation, f, batch=batch)

    elif permutation_type == PermutationType.pairings:
        calc_permutation = partial(_paired_permutation, f, batch=batch)

    else:
        raise ValueError(f'Got unexpected value for permutation_type: {permutation_type}')

    if isinstance(random_state, int):
        rng = np.random.default_rng(seed=random_state)

    elif isinstance(random_state, np.random.Generator):
        rng = random_state

    elif random_state is None:
        rng = np.random.default_rng()

    else:
        raise ValueError(f'Got unexpected value for random_state: {random_state}')

    seeds = rng.integers(0, INT_MAX, iterations)
    sample_statistic = delayed(calc_permutation)(args, shuffle=False)
    jobs = chain([sample_statistic],
                 (delayed(calc_permutation)(args, seed) for seed in seeds))

    permutation_statistics = parallel(jobs)
    sample_statistic = permutation_statistics[0]
    permutation_statistics = np.array(permutation_statistics[1:])
    if len(sample_statistic) == 1:
        sample_statistic = sample_statistic[0]
        sample_delta = sample_statistic
        permutation_deltas = permutation_statistics

    elif len(args) > 2 or alternative == Alternative.two_sided:
        sample_delta = np.max(sample_statistic) - np.min(sample_statistic)
        permutation_deltas = np.max(permutation_statistics, axis=1) - np.min(permutation_statistics, axis=1)

    elif alternative == Alternative.greater:
        sample_delta = sample_statistic[0] - sample_statistic[1]
        permutation_deltas = permutation_statistics[:, 0] - permutation_statistics[:, 1]

    elif alternative == Alternative.less:
        sample_delta = sample_statistic[1] - sample_statistic[0]
        permutation_deltas = permutation_statistics[:, 1] - permutation_statistics[:, 0]

    else:
        raise ValueError(f'Got unexpected value for alternative: {alternative}')

    p_value = np.sum(sample_delta <= permutation_deltas) / iterations

    return TestResult(statistic=sample_statistic, p_value=p_value)


# Evaluate a single permutation in an independent-style permutation test.
def _ind_permutation(f, args, seed=0, shuffle=True, batch=False):
    rng = np.random.default_rng(seed)
    if shuffle:
        orig_shape = [x.shape for x in args]
        args = [arg.flatten() for arg in args]
        orig_len = [len(arg) for arg in args]
        args = np.concatenate(args)
        rng.shuffle(args)
        new_args = []
        for length, shape in zip(orig_len, orig_shape):
            new_arg = args[:length]
            new_arg.shape = shape
            args = args[length:]
            new_args.append(new_arg)

        args = new_args

    if hasattr(f, '_accepts_random_state'):
        f = partial(f, random_state=rng)

    if batch:
        statistics = f(args)

    else:
        statistics = tuple([f(x) for x in args])

    return statistics


# Evaluate a single permutation in a pairings-style permutation test.
def _paired_permutation(f, args, seed=0, shuffle=True, batch=False):
    rng = np.random.default_rng(seed)
    if shuffle:
        rng = np.random.default_rng(seed)
        for i in range(len(args)):
            args[i] = np.copy(args[i])
            rng.shuffle(args[i])

    if hasattr(f, '_accepts_random_state'):
        f = partial(f, random_state=rng)

    if batch:
        statistics = f(args)

    else:
        statistics = tuple([f(x) for x in args])

    return statistics


def _samples_permutation(f, args, seed=0, shuffle=True, batch=False):
    rng = np.random.default_rng(seed)
    if shuffle:
        new_args = []
        for arg in zip(*args):
            arg = np.array(arg)
            rng.shuffle(arg)
            new_args.append(arg)

        args = np.array(new_args)

    if hasattr(f, '_accepts_random_state'):
        f = partial(f, random_state=rng)

    if batch:
        statistics = f(args)

    else:
        statistics = tuple([f(x) for x in args])

    return statistics


def accepts_random_state(f):
    """
    Used as a decorator, this function indicates to 'permutation_test'
    that the given function, 'f', has a 'random_state' parameter.
    The function must be able to accept an instance of
    numpy.random.Generator as an argument to random_state.

    """

    f._accepts_random_state = True
    return f


def iqr(x, axis=None):
    """
    Calculate the interquartile range of the given ndarray.

    x: The ndarray to calculate the IQR of.
    axis: The axis along 'x' to calculate the IQR. By default
          a flattened version of the array is used.

    Returns:
      The IQR of 'x'. If 'x' is a 1D array or axis is None, this will be
      a scalar value. If 'x' has more than 1 dimension and axis is not
      None, it will be an array instead.

    """

    q1, q3 = np.quantile(x, [.25, .75], axis=axis)
    return q3 - q1


test_mean = partial(permutation_test, np.mean)
test_gmean = partial(permutation_test, stats.mstats.gmean)
test_hmean = partial(permutation_test, stats.hmean)
test_variance = partial(permutation_test, np.var)
test_skewness = partial(permutation_test, stats.skew)
test_kurtosis = partial(permutation_test, stats.kurtosis)
test_median = partial(permutation_test, np.median)
test_mad = partial(permutation_test, stats.median_abs_deviation)
test_iqr = partial(permutation_test, iqr)
test_medcouple = partial(permutation_test, medcouple)
test_mode = partial(permutation_test, stats.mode)


def test_tail_weight(a, b, *args, side=DistSide.both, **kwargs):
    """
    Conduct a permutation test of the tail weights of datasets a, b, c, etc.

    Args:
      a: An ndarray representing the first dataset under study.
      b: An ndarray representing the second dataset under study.
      *args: ndarrays representing subsequent datasets in the study.
      side: See the documentation of 'tail_weight' for the meaning of
            this parameter.
      **kwargs: Additional keyword arguments will be passed through to
                'permutation_test'.

    Returns:
      An instance of 'TestResult' where the statistic is the tail weight
      of each ndarray passed to 'test_tail_weight'.

    """

    return permutation_test(partial(tail_weight, side=side), a, b, *args, **kwargs)


def test_trimmed_mean(a, b, *args, proportiontocut=0.1, **kwargs):
    """
    Conduct a permutation test of the trimmed means of datasets a, b, c, etc.

    Args:
      a: An ndarray representing the first dataset under study.
      b: An ndarray representing the second dataset under study.
      *args: ndarrays representing subsequent datasets in the study.
      proportiontocut: See the documentation of 'scipy.stats.trim_mean'
                       for the meaning of this parameter.
      **kwargs: Additional keyword arguments will be passed through to
                'permutation_test'.

    Returns:
      An instance of 'TestResult' where the statistic is the trimmed mean
      of each ndarray passed to 'test_trimmed_mean'.

    """

    return permutation_test(partial(stats.trim_mean, proportiontocut=proportiontocut),
                            a,
                            b,
                            *args,
                            **kwargs)


# Calculate Silhouette Coefficient over axis 1 of a 2D ndarray.
def _silhouette_coeff(X, metric='euclidean'):
    y = []
    for i, a in enumerate(X):
        y.append(np.full(len(a), i))

    y = np.concatenate(y)
    X = np.concatenate(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return [silhouette_score(X, y, metric=metric)]


def test_silhouette(a, b, *args, metric='euclidean', **kwargs):
    """
    Conduct a permutation test of the Silhouette Coefficient of datasets
    a, b, c, etc.

    Args:
      a: An ndarray representing the first dataset under study.
      b: An ndarray representing the second dataset under study.
      *args: ndarrays representing subsequent datasets in the study.
      metric: See the documentation of 'sklearn.metrics.silhouette_score'
              for the meaning of this parameter.
      **kwargs: Additional keyword arguments will be passed through to
                'permutation_test'.

    Returns:
      An instance of 'TestResult' where the statistic is the Silhouette
      Coefficient calculated on the ndarrays passed to 'test_silhouette'.

    """

    return permutation_test(partial(_silhouette_coeff, metric=metric),
                            a,
                            b,
                            *args,
                            batch=True,
                            **kwargs)


# Calculate mutual information on discrete datasets.
def _mutual_info_discrete(data, average_method='arithmetic'):
    return [adjusted_mutual_info_score(data[0], data[1], average_method=average_method)]


# Calculate mutual information on continuous datasets.
@accepts_random_state
def _mutual_info_continuous(data, discrete_features=False, n_neighbors=3, random_state=None):
    x = data[0].reshape(-1, 1)
    y = data[1]
    return mutual_info_regression(x, y,
                                  discrete_features=discrete_features,
                                  n_neighbors=n_neighbors,
                                  random_state=random_state)


def test_mutual_info(a, b,
                     a_discrete=True,
                     b_discrete=True,
                     average_method='arithmetic',
                     n_neighbors=3,
                     **kwargs):
    if a_discrete and b_discrete:
        f = partial(_mutual_info, average_method=average_method)

    elif a_discrete:
        f = partial(_mutual_info_continuous, discrete_features=True, n_neighbors=n_neighbors)

    elif b_discrete:
        c = a
        a = b
        b = c
        f = partial(_mutual_info_continuous, discrete_features=True, n_neighbors=n_neighbors)

    else:
        f = partial(_mutual_info_continuous, discrete_features=False, n_neighbors=n_neighbors)

    return permutation_test(f, a, b,
                            batch=True,
                            permutation_type=PermutationType.pairings,
                            **kwargs)
