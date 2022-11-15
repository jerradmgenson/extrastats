"""
Test cases for extrastats.py

Copyright 2022 Jerrad Michael Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import scipy as sp
from sklearn.metrics import mean_squared_error

import extrastats as es


class TestPermutationTest(unittest.TestCase):
    """
    Test cases for es.permutation_test

    """

    def test_uniform_random_means_are_equal(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = es.permutation_test(np.mean, a, b, random_state=rng)

        self.assertAlmostEqual(test_result.pvalue, 0.113)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 50.58082181)

    def test_uniform_random_means_are_not_equal(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(5, 105, 10000)
        test_result = es.permutation_test(np.mean, a, b, random_state=rng)

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 55.58082181)

    def test_uniform_random_mean_is_less(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(5, 105, 10000)
        test_result = es.permutation_test(
            np.mean, a, b, random_state=rng, alternative=es.Alternative.less
        )

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 55.58082181)

    def test_uniform_random_mean_is_greater(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(5, 105, 10000)
        test_result = es.permutation_test(
            np.mean, a, b, random_state=rng, alternative=es.Alternative.greater
        )

        self.assertAlmostEqual(test_result.pvalue, 1)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 55.58082181)

    def test_uniform_random_variance_is_equal(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(100, 200, 10000)
        test_result = es.permutation_test(
            np.var, a, b, random_state=rng, alternative=es.Alternative.two_sided
        )

        self.assertAlmostEqual(test_result.pvalue, 0.718)
        self.assertAlmostEqual(test_result.statistic[0], 835.50152489)
        self.assertAlmostEqual(test_result.statistic[1], 819.54902091)

    def test_uniform_random_variance_is_not_equal(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 200, 10000)
        test_result = es.permutation_test(
            np.var, a, b, random_state=rng, alternative=es.Alternative.two_sided
        )

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 835.50152489)
        self.assertAlmostEqual(test_result.statistic[1], 3278.19608364)

    def test_uniform_random_variance_is_not_less(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(100, 200, 10000)
        test_result = es.permutation_test(
            np.var, a, b, random_state=rng, alternative=es.Alternative.less
        )

        self.assertAlmostEqual(test_result.pvalue, 0.628)
        self.assertAlmostEqual(test_result.statistic[0], 835.50152489)
        self.assertAlmostEqual(test_result.statistic[1], 819.54902091)

    def test_with_two_cpus(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = es.permutation_test(np.mean, a, b, random_state=rng, n_jobs=2)

        self.assertAlmostEqual(test_result.pvalue, 0.113)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 50.58082181)

    def test_with_all_cpus(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = es.permutation_test(np.mean, a, b, random_state=rng, n_jobs=-1)

        self.assertAlmostEqual(test_result.pvalue, 0.113)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 50.58082181)

    def test_500_iterations(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = es.permutation_test(
            np.mean, a, b, random_state=rng, iterations=500
        )

        self.assertAlmostEqual(test_result.pvalue, 0.112)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 50.58082181)

    def test_10000_iterations(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = es.permutation_test(
            np.mean, a, b, random_state=rng, iterations=10000
        )

        self.assertAlmostEqual(test_result.pvalue, 0.1146)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 50.58082181)

    def test_gaussian_means_are_equal(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 10, 10000)
        b = rng.normal(50, 100, 10000)
        test_result = es.permutation_test(np.mean, a, b, random_state=rng)

        self.assertAlmostEqual(test_result.pvalue, 0.815)
        self.assertAlmostEqual(test_result.statistic[0], 50.06311887)
        self.assertAlmostEqual(test_result.statistic[1], 50.30509898)

    def test_gaussian_variance_is_greater(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 10, 10000)
        b = rng.normal(50, 9, 10000)
        test_result = es.permutation_test(
            np.var, a, b, random_state=rng, alternative=es.Alternative.greater
        )

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 99.61574236)
        self.assertAlmostEqual(test_result.statistic[1], 80.02988837)

    def test_bimodal_medians_are_equal(self):
        rng = np.random.default_rng(0)
        a1 = rng.normal(50, 10, 5000)
        a2 = rng.normal(100, 20, 5000)
        a = np.concatenate([a1, a2])
        b1 = rng.normal(50, 10, 5000)
        b2 = rng.normal(100, 20, 5000)
        b = np.concatenate([b1, b2])
        test_result = es.permutation_test(np.median, a, b, random_state=rng)

        self.assertAlmostEqual(test_result.pvalue, 0.997)
        self.assertAlmostEqual(test_result.statistic[0], 66.8770881)
        self.assertAlmostEqual(test_result.statistic[1], 66.87896185)

    def test_bimodal_medians_are_not_equal(self):
        rng = np.random.default_rng(0)
        a1 = rng.normal(50, 10, 5000)
        a2 = rng.normal(100, 20, 5000)
        a = np.concatenate([a1, a2])
        b1 = rng.normal(50, 10, 2500)
        b2 = rng.normal(100, 20, 7500)
        b = np.concatenate([b1, b2])
        test_result = es.permutation_test(np.median, a, b, random_state=rng)

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 66.8770881)
        self.assertAlmostEqual(test_result.statistic[1], 91.09403305)

    def test_bimodal_mad_is_equal(self):
        rng = np.random.default_rng(0)
        a1 = rng.normal(50, 10, 5000)
        a2 = rng.normal(100, 10, 5000)
        a = np.concatenate([a1, a2])
        b1 = rng.normal(50, 10, 5000)
        b2 = rng.normal(100, 10, 5000)
        b = np.concatenate([b1, b2])
        test_result = es.permutation_test(
            sp.stats.median_abs_deviation,
            a,
            b,
            random_state=rng,
            permutation_type=es.PermutationType.independent,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.19)
        self.assertAlmostEqual(test_result.statistic[0], 25.23555214)
        self.assertAlmostEqual(test_result.statistic[1], 24.99953729)

    def test_longtailed_median_is_less(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 100, 20000)
        a = a[a >= 0]
        b = rng.normal(5, 100, 20000)
        b = b[b >= 5]
        test_result = es.permutation_test(
            np.median,
            a,
            b,
            random_state=rng,
            alternative=es.Alternative.less,
            permutation_type=es.PermutationType.independent,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.008)
        self.assertAlmostEqual(test_result.statistic[0], 69.00820654696294)
        self.assertAlmostEqual(test_result.statistic[1], 72.06607121240145)

    def test_longtailed_median_is_not_greater(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 100, 20000)
        a = a[a >= 0]
        b = rng.normal(5, 100, 20000)
        b = b[b >= 5]
        test_result = es.permutation_test(
            np.median,
            a,
            b,
            random_state=rng,
            alternative=es.Alternative.greater,
            permutation_type=es.PermutationType.independent,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.992)
        self.assertAlmostEqual(test_result.statistic[0], 69.00820654696294)
        self.assertAlmostEqual(test_result.statistic[1], 72.06607121240145)

    def test_longtailed_iqr_is_equal(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 100, 20000)
        a = a[a >= 0]
        b = rng.normal(1000, 100, 20000)
        b = b[b <= 1000]
        test_result = es.permutation_test(
            es.iqr,
            a,
            b,
            random_state=rng,
            permutation_type=es.PermutationType.independent,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.18)
        self.assertAlmostEqual(test_result.statistic[0], 81.427687223)
        self.assertAlmostEqual(test_result.statistic[1], 84.514307719)

    def test_longtailed_iqr_is_greater(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 110, 20000)
        a = a[a >= 0]
        b = rng.normal(1000, 100, 20000)
        b = b[b <= 1000]
        test_result = es.permutation_test(
            es.iqr,
            a,
            b,
            random_state=rng,
            alternative=es.Alternative.greater,
            permutation_type=es.PermutationType.independent,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.014)
        self.assertAlmostEqual(test_result.statistic[0], 89.5704559453)
        self.assertAlmostEqual(test_result.statistic[1], 84.514307719)

    def test_random_numbers_are_linearly_uncorrelated1(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = es.permutation_test(
            lambda a, b: sp.stats.pearsonr(a, b)[0],
            a,
            b,
            random_state=rng,
            batch=True,
            permutation_type=es.PermutationType.pairings,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.198)
        self.assertAlmostEqual(test_result.statistic, 0.008244083)

    def test_random_numbers_are_linearly_uncorrelated2(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = es.permutation_test(
            lambda a, b: sp.stats.pearsonr(a, b)[0],
            a,
            b,
            random_state=rng,
            batch=True,
            permutation_type=es.PermutationType.pairings,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.491)
        self.assertAlmostEqual(test_result.statistic, 0.00027762)

    def test_random_numbers_are_linearly_uncorrelated3(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = es.permutation_test(
            lambda a, b: sp.stats.pearsonr(a, b)[0],
            a,
            b,
            random_state=rng,
            batch=True,
            permutation_type=es.PermutationType.pairings,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.877)
        self.assertAlmostEqual(test_result.statistic, -0.01024487)

    def test_random_numbers_are_monotonically_uncorrelated1(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = es.permutation_test(
            lambda a, b: sp.stats.kendalltau(a, b)[0],
            a,
            b,
            random_state=rng,
            batch=True,
            permutation_type=es.PermutationType.pairings,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.194)
        self.assertAlmostEqual(test_result.statistic, 0.00557355)

    def test_random_numbers_are_monotonically_uncorrelated2(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = es.permutation_test(
            lambda a, b: sp.stats.kendalltau(a, b)[0],
            a,
            b,
            random_state=rng,
            batch=True,
            permutation_type=es.PermutationType.pairings,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.491)
        self.assertAlmostEqual(test_result.statistic, 0.00020594)

    def test_random_numbers_are_monotonically_uncorrelated3(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = es.permutation_test(
            lambda a, b: sp.stats.kendalltau(a, b)[0],
            a,
            b,
            random_state=rng,
            batch=True,
            permutation_type=es.PermutationType.pairings,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.878)
        self.assertAlmostEqual(test_result.statistic, -0.00681552)

    def test_linear_function_exhibits_linear_correlation(self):
        rng = np.random.default_rng(2)
        a = np.arange(0, 10000)
        b = a * 10
        test_result = es.permutation_test(
            lambda a, b: sp.stats.pearsonr(a, b)[0],
            a,
            b,
            random_state=rng,
            batch=True,
            permutation_type=es.PermutationType.pairings,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.0)
        self.assertAlmostEqual(test_result.statistic, 1.0)

    def test_linear_function_exhibits_monotonically_correlation(self):
        rng = np.random.default_rng(2)
        a = np.arange(0, 10000)
        b = a * 10
        test_result = es.permutation_test(
            lambda a, b: sp.stats.kendalltau(a, b)[0],
            a,
            b,
            random_state=rng,
            batch=True,
            permutation_type=es.PermutationType.pairings,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.0)
        self.assertAlmostEqual(test_result.statistic, 1.0)

    def test_linear_function_exhibits_inverse_linear_correlation(self):
        rng = np.random.default_rng(2)
        a = np.arange(0, 10000)
        b = a * -10
        test_result = es.permutation_test(
            lambda a, b: sp.stats.pearsonr(a, b)[0],
            a,
            b,
            random_state=rng,
            batch=True,
            permutation_type=es.PermutationType.pairings,
            less_is_more=True,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.0)
        self.assertAlmostEqual(test_result.statistic, -1.0)

    def test_exponential_function_exhibits_monotonic_correlation(self):
        rng = np.random.default_rng(2)
        a = np.arange(0, 100)
        b = np.exp(a)
        test_result = es.permutation_test(
            lambda a, b: sp.stats.kendalltau(a, b)[0],
            a,
            b,
            random_state=rng,
            batch=True,
            permutation_type=es.PermutationType.pairings,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.0)
        self.assertAlmostEqual(test_result.statistic, 1.0)

    def test_sinusoidal_function_is_linearly_uncorrelated(self):
        rng = np.random.default_rng(2)
        a = np.arange(0, 10000)
        b = np.sin(a)
        test_result = es.permutation_test(
            lambda a, b: sp.stats.pearsonr(a, b)[0],
            a,
            b,
            random_state=rng,
            batch=True,
            permutation_type=es.PermutationType.pairings,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.498)
        self.assertAlmostEqual(test_result.statistic, 2.67353702e-05)

    def test_sinusoidal_function_is_monotonically_uncorrelated(self):
        rng = np.random.default_rng(2)
        a = np.arange(0, 10000)
        b = np.sin(a)
        test_result = es.permutation_test(
            lambda a, b: sp.stats.kendalltau(a, b)[0],
            a,
            b,
            random_state=rng,
            batch=True,
            permutation_type=es.PermutationType.pairings,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.5)
        self.assertAlmostEqual(test_result.statistic, 2.2362236e-05)

    def test_paired_means_are_equal1(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        treatment = rng.normal(0, 1, 10000)
        b = a + treatment
        test_result = es.permutation_test(
            np.mean,
            a,
            b,
            random_state=rng,
            n_jobs=-1,
            permutation_type=es.PermutationType.samples,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.666)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 49.94551775)

    def test_paired_means_are_equal2(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(0, 1000, 10000)
        treatment = rng.normal(0, 5, 10000)
        b = a + treatment
        test_result = es.permutation_test(
            np.mean,
            a,
            b,
            random_state=rng,
            n_jobs=-1,
            permutation_type=es.PermutationType.samples,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.205)
        self.assertAlmostEqual(test_result.statistic[0], 502.044169231)
        self.assertAlmostEqual(test_result.statistic[1], 501.98310374)

    def test_paired_means_are_equal3(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-500, 500, 10000)
        treatment = rng.normal(0, 10, 10000)
        b = a + treatment
        test_result = es.permutation_test(
            np.mean,
            a,
            b,
            random_state=rng,
            n_jobs=-1,
            permutation_type=es.PermutationType.samples,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.854)
        self.assertAlmostEqual(test_result.statistic[0], 0.01720119)
        self.assertAlmostEqual(test_result.statistic[1], 0.03449550)

    def test_paired_means_are_not_equal(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-500, 500, 10000)
        treatment = rng.normal(5, 10, 10000)
        b = a + treatment
        test_result = es.permutation_test(
            np.mean,
            a,
            b,
            random_state=rng,
            n_jobs=-1,
            permutation_type=es.PermutationType.samples,
        )

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 0.01720119)
        self.assertAlmostEqual(test_result.statistic[1], 5.0344955)

    def test_paired_mean_is_less(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-500, 500, 10000)
        treatment = rng.normal(5, 10, 10000)
        b = a + treatment
        test_result = es.permutation_test(
            np.mean,
            a,
            b,
            random_state=rng,
            n_jobs=-1,
            alternative=es.Alternative.less,
            permutation_type=es.PermutationType.samples,
        )

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 0.01720119)
        self.assertAlmostEqual(test_result.statistic[1], 5.0344955)

    def test_paired_mean_is_not_greater(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-500, 500, 10000)
        treatment = rng.normal(5, 10, 10000)
        b = a + treatment
        test_result = es.permutation_test(
            np.mean,
            a,
            b,
            random_state=rng,
            n_jobs=-1,
            alternative=es.Alternative.greater,
            permutation_type=es.PermutationType.samples,
        )

        self.assertAlmostEqual(test_result.pvalue, 1.0)
        self.assertAlmostEqual(test_result.statistic[0], 0.01720119)
        self.assertAlmostEqual(test_result.statistic[1], 5.0344955)

    def test_paired_variance_is_not_less(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-500, 500, 10000)
        treatment = rng.normal(5, 10, 10000)
        b = a + treatment
        test_result = es.permutation_test(
            np.var,
            a,
            b,
            random_state=rng,
            n_jobs=-1,
            alternative=es.Alternative.less,
            permutation_type=es.PermutationType.samples,
        )

        self.assertAlmostEqual(test_result.pvalue, 0.133)
        self.assertAlmostEqual(test_result.statistic[0], 84750.7259122)
        self.assertAlmostEqual(test_result.statistic[1], 84826.54957818)

    def test_paired_variance_is_greater(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-500, 500, 10000)
        b = np.where(a > 0, a - 0.1, a + 0.1)
        test_result = es.permutation_test(
            np.var,
            a,
            b,
            random_state=rng,
            n_jobs=-1,
            alternative=es.Alternative.greater,
            permutation_type=es.PermutationType.samples,
        )

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 84750.7259122)
        self.assertAlmostEqual(test_result.statistic[1], 84700.17749948)

    def test_mean_squared_error(self):
        rng = np.random.default_rng(101)
        a = np.arange(100) * 5 + 7
        test_result = es.permutation_test(
            lambda x, y: -1 * mean_squared_error(x, y),
            a,
            a,
            permutation_type=es.PermutationType.pairings,
            batch=True,
            random_state=rng,
        )

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic, 0)


class TestIQR(unittest.TestCase):
    """
    Test cases for es.iqr

    """

    def test_s4(self):
        a = np.array([3, 1, 2, 4])
        iqr = es.iqr(a)
        self.assertEqual(iqr, 2)

    def test_s5(self):
        a = np.array([3, 5, 4, 2, 1])
        iqr = es.iqr(a)
        self.assertEqual(iqr, 3)

    def test_s6(self):
        a = np.array([5, 6, 2, 3, 1, 4])
        iqr = es.iqr(a)
        self.assertEqual(iqr, 3)

    def test_s7(self):
        a = np.array([4, 3, 1, 7, 5, 6, 2])
        iqr = es.iqr(a)
        self.assertEqual(iqr, 4)

    def test_large_array(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100000, 100000000)
        iqr = es.iqr(a)
        self.assertAlmostEqual(iqr, 49998.53616906)


class TestTailWeight(unittest.TestCase):
    """
    Tests for extratest.tail_weight

    """

    def test_standard_normal_left(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 1000)
        tw = es.tail_weight(a, side=es.DistSide.left)
        self.assertAlmostEqual(tw, 0.21362931)

    def test_standard_normal_right(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 1000)
        tw = es.tail_weight(a, side=es.DistSide.right)
        self.assertAlmostEqual(tw, 0.18177707)

    def test_standard_normal_both(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 1000)
        tw = es.tail_weight(a, side=es.DistSide.both)
        self.assertAlmostEqual(tw, 0.19770319)

    def test_left_skewed_dist_left(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 10, 1000)
        b = rng.uniform(40, 60, 200)
        a = np.concatenate([b, a])
        tw = es.tail_weight(a, side=es.DistSide.left)
        self.assertAlmostEqual(tw, 0.67276829)

    def test_left_skewed_dist_right(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 10, 1000)
        b = rng.uniform(40, 60, 200)
        a = np.concatenate([b, a])
        tw = es.tail_weight(a, side=es.DistSide.right)
        self.assertAlmostEqual(tw, 0.18488636)

    def test_left_skewed_dist_both(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 10, 1000)
        b = rng.uniform(40, 60, 200)
        a = np.concatenate([b, a])
        tw = es.tail_weight(a, side=es.DistSide.both)
        self.assertAlmostEqual(tw, 0.42882732)

    def test_bimodal_left1(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 5, 1000)
        b = rng.normal(100, 10, 2011)
        a = np.concatenate([a, b])
        tw = es.tail_weight(a, side=es.DistSide.left)
        self.assertAlmostEqual(tw, -0.645384209)

    def test_bimodal_right1(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 5, 1000)
        b = rng.normal(100, 10, 2011)
        a = np.concatenate([a, b])
        tw = es.tail_weight(a, side=es.DistSide.right)
        self.assertAlmostEqual(tw, 0.11993853)

    def test_bimodal_both1(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 5, 1000)
        b = rng.normal(100, 10, 2011)
        a = np.concatenate([a, b])
        tw = es.tail_weight(a, side=es.DistSide.both)
        self.assertAlmostEqual(tw, -0.26272283)

    def test_bimodal_left2(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 20, 1000)
        b = rng.normal(1000, 20, 100)
        a = np.concatenate([a, b])
        tw = es.tail_weight(a, side=es.DistSide.left)
        self.assertAlmostEqual(tw, 0.17367395)

    def test_bimodal_right2(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 20, 1000)
        b = rng.normal(1000, 20, 100)
        a = np.concatenate([a, b])
        tw = es.tail_weight(a, side=es.DistSide.right)
        self.assertAlmostEqual(tw, 0.49642536)

    def test_bimodal_both2(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 20, 1000)
        b = rng.normal(1000, 20, 100)
        a = np.concatenate([a, b])
        tw = es.tail_weight(a, side=es.DistSide.both)
        self.assertAlmostEqual(tw, 0.33504965)


class TestAcceptsRandomState(unittest.TestCase):
    """
    Tests for es.accepts_random_state

    """

    def test_has_attr(self):
        @es.accepts_random_state
        def test():
            pass

        self.assertTrue(hasattr(test, "_accepts_random_state"))


class TestMutualInfo(unittest.TestCase):
    """
    Tests for es.test_mutual_info

    """

    def test_correctly_identifies_discrete_dependent_variables(self):
        a = np.repeat(np.arange(1, 11), 100)
        b = a % 2
        test_result = es.test_mutual_info(a, b, random_state=0)
        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic, 0.46112849)

    def test_correctly_identifies_continuous_dependent_variables(self):
        a = np.repeat(np.arange(1, 11, 0.1), 10)
        b = np.sin(a)
        test_result = es.test_mutual_info(
            a, b, a_discrete=False, b_discrete=False, random_state=0
        )

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic, 4.66252284)

    def test_correctly_identifies_mixed_dependent_variables1(self):
        a = np.repeat(np.arange(1, 11), 10)
        b = np.sin(a)
        test_result = es.test_mutual_info(
            a, b, a_discrete=True, b_discrete=False, random_state=0
        )

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic, 2.34840926)

    def test_correctly_identifies_mixed_dependent_variables2(self):
        a = np.repeat(np.arange(1, 11), 10)
        b = np.sin(a)
        test_result = es.test_mutual_info(
            b, a, a_discrete=False, b_discrete=True, random_state=0
        )

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic, 2.34840926)

    def test_correctly_identifies_discrete_independent_variables(self):
        a = np.repeat(np.arange(1, 11), 100)
        b = a % 2
        rng = np.random.default_rng(0)
        rng.shuffle(b)
        test_result = es.test_mutual_info(a, b, random_state=rng)
        self.assertAlmostEqual(test_result.pvalue, 0.308)
        self.assertAlmostEqual(test_result.statistic, 0.00050649)

    def test_correctly_identifies_continuous_independent_variables(self):
        a = np.repeat(np.arange(1, 11, 0.1), 10)
        b = np.sin(a)
        rng = np.random.default_rng(0)
        rng.shuffle(b)
        test_result = es.test_mutual_info(
            a, b, a_discrete=False, b_discrete=False, random_state=rng
        )

        self.assertAlmostEqual(test_result.pvalue, 0.407)
        self.assertAlmostEqual(test_result.statistic, 0.003932137)

    def test_correctly_identifies_mixed_independent_variables1(self):
        a = np.repeat(np.arange(1, 11), 10)
        b = np.sin(a)
        rng = np.random.default_rng(0)
        rng.shuffle(b)
        test_result = es.test_mutual_info(
            a, b, a_discrete=True, b_discrete=False, random_state=0
        )

        self.assertAlmostEqual(test_result.pvalue, 1.0)
        self.assertAlmostEqual(test_result.statistic, 0.0)

    def test_correctly_identifies_mixed_independent_variables2(self):
        a = np.repeat(np.arange(1, 11), 10)
        b = np.sin(a)
        rng = np.random.default_rng(0)
        rng.shuffle(a)
        test_result = es.test_mutual_info(
            b, a, a_discrete=False, b_discrete=True, random_state=0
        )

        self.assertAlmostEqual(test_result.pvalue, 1.0)
        self.assertAlmostEqual(test_result.statistic, 0.0)


class TestStandardError(unittest.TestCase):
    """
    Test cases for es.standard_error

    """

    def test_standard_error_of_natural_means(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 1000, 30)
        se = es.standard_error(np.mean, x, random_state=rng)
        self.assertAlmostEqual(se, 55.2180022977)

    def test_standard_error_of_integer_means(self):
        rng = np.random.default_rng(1)
        x = rng.integers(-1000, 1000, 300)
        se = es.standard_error(np.mean, x, random_state=rng)
        self.assertAlmostEqual(se, 33.1126733713)

    def test_standard_error_of_real_means(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(-1000, 1000, 300)
        se = es.standard_error(np.mean, x, random_state=rng)
        self.assertAlmostEqual(se, 33.366064398)


class TestAdjustedBoxplot(unittest.TestCase):
    """
    Test cases for es.adjusted_boxplot

    """

    def test_uniform_distribution_ndarray(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-100, 100, 1000)
        low, high = es.adjusted_boxplot(x)
        y = x[np.logical_or(x < low, x > high)]
        self.assertEqual(len(y), 0)

    def test_uniform_distribution_ndarray_with_one_outlier(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.uniform(-100, 100, 1000), np.full(1, 10000)])

        low, high = es.adjusted_boxplot(x)
        y = x[np.logical_or(x < low, x > high)]
        self.assertEqual(y, [10000])

    def test_uniform_distribution_ndarray_with_two_outliers(self):
        rng = np.random.default_rng(0)
        x = np.concatenate(
            [rng.uniform(-100, 100, 1000), np.full(1, 10000), np.full(1, -10000)]
        )

        low, high = es.adjusted_boxplot(x)
        y = x[np.logical_or(x < low, x > high)]
        self.assertTrue((y == [10000, -10000]).all())

    def test_normal_distribution_with_15_outliers(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 1000)
        low, high = es.adjusted_boxplot(x)
        y = x[np.logical_or(x < low, x > high)]
        self.assertEqual(len(y), 15)

    def test_normal_distribution_with_8_outliers(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 100, 1000)
        low, high = es.adjusted_boxplot(x)
        y = x[np.logical_or(x < low, x > high)]
        self.assertEqual(len(y), 8)

    def test_normal_distribution_with_high_k(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 1000)
        low, high = es.adjusted_boxplot(x, k=3)
        y = x[np.logical_or(x < low, x > high)]
        self.assertEqual(len(y), 0)

    def test_normal_distribution_with_low_k(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 1000)
        low, high = es.adjusted_boxplot(x, k=1)
        y = x[np.logical_or(x < low, x > high)]
        self.assertEqual(len(y), 44)

    def test_right_tailed_distribution(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(0, 1, 1000), rng.uniform(1, 5, 100)])

        low, high = es.adjusted_boxplot(x)
        y = x[np.logical_or(x < low, x > high)]
        self.assertAlmostEqual(low, -2.10054411)
        self.assertAlmostEqual(high, 4.37154061)
        self.assertEqual(len(y), 32)

    def test_left_tailed_distribution(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(0, 1, 1000), rng.uniform(-5, -1, 100)])

        low, high = es.adjusted_boxplot(x)
        y = x[np.logical_or(x < low, x > high)]
        self.assertAlmostEqual(low, -3.62692345)
        self.assertAlmostEqual(high, 2.48356375)
        self.assertEqual(len(y), 44)

    @patch("extrastats.medcouple", MagicMock(return_value=np.nan))
    def test_medcouple_nan_raises_exception(self):
        rng = np.random.default_rng(510030955)
        x = rng.uniform(-1000000, 1000000, 1000)
        with self.assertRaises(es.MedcoupleError):
            es.adjusted_boxplot(x)

    @patch("extrastats.medcouple", MagicMock(return_value=np.nan))
    def test_medcouple_nan_logs_warning(self):
        rng = np.random.default_rng(510030955)
        x = rng.uniform(-1000000, 1000000, 1000)
        with self.assertLogs(level="WARNING"):
            low, high = es.adjusted_boxplot(x, raise_medcouple_error=False)

        self.assertTrue(np.isfinite(low))
        self.assertTrue(np.isfinite(high))
        self.assertLess(low, high)


class TestGCV(unittest.TestCase):
    """
    Test cases for es.gcv

    """

    def test_even_integers(self):
        self.assertAlmostEqual(es.gcv(np.arange(2, 21, 2)), 0.78859946)

    def test_floats1(self):
        self.assertAlmostEqual(es.gcv([0.1, 0.2, 0.4, 0.8]), 0.907276639)

    def test_floats2(self):
        x = [
            6.70497622,
            3.71651977,
            1.00910303,
            0.87033352,
            2.52396456,
            4.03251592,
            5.2215431,
            2.61256494,
            3.31793736,
            3.90597434,
        ]
        self.assertAlmostEqual(es.gcv(x), 0.692575156)

    def test_array_with_cardinality_0(self):
        x = np.array([])
        result = es.gcv(x)
        self.assertTrue(np.isnan(result))

    def test_list(self):
        x = [0.1, 0.2, 0.4, 0.8]
        result = es.gcv(x)
        self.assertAlmostEqual(result, 0.907276639)


class TestSumProb(unittest.TestCase):
    """
    Test cases for es.sum_prob

    """

    def test_commutativity(self):
        """
        Test that es.sum_prob satisfies the commutative property

        """

        rng = np.random.default_rng(0)
        for _ in range(10000):
            p = rng.uniform(size=2)
            sum1 = es.sum_prob(p)
            p = np.flip(p)
            sum2 = es.sum_prob(p)
            self.assertAlmostEqual(sum1, sum2)

    def test_associativity(self):
        """
        Test that es.sum_prob satisfies the associative property

        """

        rng = np.random.default_rng(1)
        for _ in range(10000):
            n = rng.integers(3, 101)
            p = rng.uniform(size=n)
            sum1 = es.sum_prob(p)
            rng.shuffle(p)
            sum2 = es.sum_prob(p)
            self.assertAlmostEqual(sum1, sum2)

    def test_probability_boundaries(self):
        """
        Test that es.sum_prob results satisfy 0<=p<=1

        """

        rng = np.random.default_rng(2)
        for _ in range(10000):
            n = rng.integers(1, 101)
            p = rng.uniform(size=n)
            result = es.sum_prob(p)
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 1)

    def test_array_with_cardinality_0(self):
        """
        Test that es.sum_prob is correct for an array of cardinality 0

        """

        result = es.sum_prob(np.array([]))
        self.assertEqual(result, 0.0)

    def test_array_with_cardinality_1(self):
        """
        Test that es.sum_prob is correct for an array of cardinality 1

        """

        p = np.array([0.5])
        result = es.sum_prob(p)
        self.assertAlmostEqual(result, p[0])

    def test_array_with_cardinality_2(self):
        """
        Test that es.sum_prob is correct for an array of cardinality 2

        """

        p = np.array([0.5, 0.5])
        result = es.sum_prob(p)
        self.assertAlmostEqual(result, 0.75)

    def test_array_with_cardinality_3(self):
        """
        Test that es.sum_prob is correct for an array of cardinality 3

        """

        p = np.array([0.5, 0.5, 0.5])
        result = es.sum_prob(p)
        self.assertAlmostEqual(result, 0.875)

    def test_list(self):
        """
        Test es.sum_prob with a list

        """

        p = [0.5, 0.5, 0.5]
        result = es.sum_prob(p)
        self.assertAlmostEqual(result, 0.875)

    def test_tuple(self):
        """
        Test es.sum_prob with a tuple

        """

        p = (0.5, 0.5, 0.5)
        result = es.sum_prob(p)
        self.assertAlmostEqual(result, 0.875)

    def test_against_simulation(self):
        """
        Test the calculations of es.sum_prob vs simulated calculations

        """

        rng = np.random.default_rng(3)

        def sim(p):
            trials = 10000
            success = 0
            for _ in range(trials):
                rand_probs = rng.uniform(size=len(p))
                if (p > rand_probs).any():
                    success += 1

            return success / trials

        for _ in range(100):
            n = rng.integers(2, 101)
            p = rng.uniform(size=n)
            result1 = round(es.sum_prob(p), 2)
            result2 = round(sim(p), 2)
            self.assertTrue(abs(result1 - result2) < 0.021)


class TestHVar(unittest.TestCase):
    """
    Test cases for es.hvar

    """

    def test_valid_input1(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(1, 100, 5)
        result = es.hvar(x)
        self.assertAlmostEqual(result, 15.173267629321453)

    def test_valid_input2(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(1, 100, 10)
        result = es.hvar(x)
        self.assertAlmostEqual(result, 103.05631937969912)

    def test_valid_input3(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(100, 1000, 5)
        result = es.hvar(x)
        self.assertAlmostEqual(result, 7485.401157374311)

    def test_array_with_cardinality_0(self):
        x = np.array([])
        result = es.hvar(x)
        self.assertTrue(np.isnan(result))

    def test_array_with_cardinality_1(self):
        x = np.array([10])
        result = es.hvar(x)
        self.assertEqual(result, 0)

    def test_array_with_cardinality_2(self):
        x = np.array([10, 20])
        result = es.hvar(x)
        self.assertAlmostEqual(result, 9.876543209876539)

    def test_list(self):
        x = [10, 20]
        result = es.hvar(x)
        self.assertAlmostEqual(result, 9.876543209876539)


if __name__ == "__main__":
    unittest.main()
