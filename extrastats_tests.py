"""
Test cases for extrastats.py

Copyright 2022 Jerrad Michael Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import unittest

import numpy as np
import scipy as sp

import extrastats


class TestPermutationTest(unittest.TestCase):
    """
    Test cases for extrastats.permutation_test

    """

    def test_uniform_random_means_are_equal(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng)

        self.assertAlmostEqual(test_result.pvalue, 0.113)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 50.58082181)

    def test_uniform_random_means_are_not_equal(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(5, 105, 10000)
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng)

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 55.58082181)

    def test_uniform_random_mean_is_less(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(5, 105, 10000)
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  alternative=extrastats.Alternative.less)

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 55.58082181)

    def test_uniform_random_mean_is_greater(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(5, 105, 10000)
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  alternative=extrastats.Alternative.greater)

        self.assertAlmostEqual(test_result.pvalue, 1)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 55.58082181)

    def test_uniform_random_variance_is_equal(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(100, 200, 10000)
        test_result = extrastats.permutation_test(np.var, a, b,
                                                  random_state=rng,
                                                  alternative=extrastats.Alternative.two_sided)

        self.assertAlmostEqual(test_result.pvalue, 0.718)
        self.assertAlmostEqual(test_result.statistic[0], 835.50152489)
        self.assertAlmostEqual(test_result.statistic[1], 819.54902091)

    def test_uniform_random_variance_is_not_equal(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 200, 10000)
        test_result = extrastats.permutation_test(np.var, a, b,
                                                  random_state=rng,
                                                  alternative=extrastats.Alternative.two_sided)

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 835.50152489)
        self.assertAlmostEqual(test_result.statistic[1], 3278.19608364)

    def test_uniform_random_variance_is_not_less(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(100, 200, 10000)
        test_result = extrastats.permutation_test(np.var, a, b,
                                                  random_state=rng,
                                                  alternative=extrastats.Alternative.less)

        self.assertAlmostEqual(test_result.pvalue, 0.628)
        self.assertAlmostEqual(test_result.statistic[0], 835.50152489)
        self.assertAlmostEqual(test_result.statistic[1], 819.54902091)

    def test_with_two_cpus(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  n_jobs=2)

        self.assertAlmostEqual(test_result.pvalue, 0.113)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 50.58082181)

    def test_with_all_cpus(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  n_jobs=-1)

        self.assertAlmostEqual(test_result.pvalue, 0.113)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 50.58082181)

    def test_500_iterations(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  iterations=500)

        self.assertAlmostEqual(test_result.pvalue, 0.112)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 50.58082181)

    def test_10000_iterations(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  iterations=10000)

        self.assertAlmostEqual(test_result.pvalue, 0.1146)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 50.58082181)

    def test_gaussian_means_are_equal(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 10, 10000)
        b = rng.normal(50, 100, 10000)
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng)

        self.assertAlmostEqual(test_result.pvalue, 0.815)
        self.assertAlmostEqual(test_result.statistic[0], 50.06311887)
        self.assertAlmostEqual(test_result.statistic[1], 50.30509898)

    def test_gaussian_variance_is_greater(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 10, 10000)
        b = rng.normal(50, 9, 10000)
        test_result = extrastats.permutation_test(np.var, a, b,
                                                  random_state=rng,
                                                  alternative=extrastats.Alternative.greater)

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
        test_result = extrastats.permutation_test(np.median, a, b,
                                                  random_state=rng)

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
        test_result = extrastats.permutation_test(np.median, a, b,
                                                  random_state=rng)

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
        test_result = extrastats.permutation_test(sp.stats.median_abs_deviation,
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  permutation_type=extrastats.PermutationType.independent)

        self.assertAlmostEqual(test_result.pvalue, 0.19)
        self.assertAlmostEqual(test_result.statistic[0], 25.23555214)
        self.assertAlmostEqual(test_result.statistic[1], 24.99953729)

    def test_longtailed_median_is_less(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 100, 20000)
        a = a[a >= 0]
        b = rng.normal(5, 100, 20000)
        b = b[b >= 5]
        test_result = extrastats.permutation_test(np.median, a, b,
                                                  random_state=rng,
                                                  alternative=extrastats.Alternative.less,
                                                  permutation_type=extrastats.PermutationType.independent)

        self.assertAlmostEqual(test_result.pvalue, 0.008)
        self.assertAlmostEqual(test_result.statistic[0], 69.00820654696294)
        self.assertAlmostEqual(test_result.statistic[1], 72.06607121240145)

    def test_longtailed_median_is_not_greater(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 100, 20000)
        a = a[a >= 0]
        b = rng.normal(5, 100, 20000)
        b = b[b >= 5]
        test_result = extrastats.permutation_test(np.median, a, b,
                                                  random_state=rng,
                                                  alternative=extrastats.Alternative.greater,
                                                  permutation_type=extrastats.PermutationType.independent)

        self.assertAlmostEqual(test_result.pvalue, 0.992)
        self.assertAlmostEqual(test_result.statistic[0], 69.00820654696294)
        self.assertAlmostEqual(test_result.statistic[1], 72.06607121240145)

    def test_longtailed_iqr_is_equal(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 100, 20000)
        a = a[a >= 0]
        b = rng.normal(1000, 100, 20000)
        b = b[b <= 1000]
        test_result = extrastats.permutation_test(extrastats.iqr, a, b,
                                                  random_state=rng,
                                                  permutation_type=extrastats.PermutationType.independent)

        self.assertAlmostEqual(test_result.pvalue, 0.18)
        self.assertAlmostEqual(test_result.statistic[0], 81.427687223)
        self.assertAlmostEqual(test_result.statistic[1], 84.514307719)

    def test_longtailed_iqr_is_greater(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 110, 20000)
        a = a[a >= 0]
        b = rng.normal(1000, 100, 20000)
        b = b[b <= 1000]
        test_result = extrastats.permutation_test(extrastats.iqr, a, b,
                                                  random_state=rng,
                                                  alternative=extrastats.Alternative.greater,
                                                  permutation_type=extrastats.PermutationType.independent)

        self.assertAlmostEqual(test_result.pvalue, 0.014)
        self.assertAlmostEqual(test_result.statistic[0], 89.5704559453)
        self.assertAlmostEqual(test_result.statistic[1], 84.514307719)

    def test_random_numbers_are_linearly_uncorrelated1(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(lambda a, b: sp.stats.pearsonr(a, b)[0],
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  batch=True,
                                                  permutation_type=extrastats.PermutationType.pairings)

        self.assertAlmostEqual(test_result.pvalue, 0.396)
        self.assertAlmostEqual(test_result.statistic, 0.008244083)

    def test_random_numbers_are_linearly_uncorrelated2(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(lambda a, b: sp.stats.pearsonr(a, b)[0],
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  batch=True,
                                                  permutation_type=extrastats.PermutationType.pairings)

        self.assertAlmostEqual(test_result.pvalue, 0.977)
        self.assertAlmostEqual(test_result.statistic, 0.00027762)

    def test_random_numbers_are_linearly_uncorrelated3(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(lambda a, b: sp.stats.pearsonr(a, b)[0],
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  batch=True,
                                                  permutation_type=extrastats.PermutationType.pairings)

        self.assertAlmostEqual(test_result.pvalue, 0.324)
        self.assertAlmostEqual(test_result.statistic, -0.01024487)

    def test_random_numbers_are_monotonically_uncorrelated1(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(lambda a, b: sp.stats.kendalltau(a, b)[0],
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  batch=True,
                                                  permutation_type=extrastats.PermutationType.pairings)

        self.assertAlmostEqual(test_result.pvalue, 0.395)
        self.assertAlmostEqual(test_result.statistic, 0.00557355)

    def test_random_numbers_are_monotonically_uncorrelated2(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(lambda a, b: sp.stats.kendalltau(a, b)[0],
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  batch=True,
                                                  permutation_type=extrastats.PermutationType.pairings)

        self.assertAlmostEqual(test_result.pvalue, 0.978)
        self.assertAlmostEqual(test_result.statistic, 0.00020594)

    def test_random_numbers_are_monotonically_uncorrelated3(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(lambda a, b: sp.stats.kendalltau(a, b)[0],
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  batch=True,
                                                  permutation_type=extrastats.PermutationType.pairings)

        self.assertAlmostEqual(test_result.pvalue, 0.327)
        self.assertAlmostEqual(test_result.statistic, -0.00681552)

    def test_linear_function_exhibits_linear_correlation(self):
        rng = np.random.default_rng(2)
        a = np.arange(0, 10000)
        b = a * 10
        test_result = extrastats.permutation_test(lambda a, b: sp.stats.pearsonr(a, b)[0],
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  batch=True,
                                                  permutation_type=extrastats.PermutationType.pairings)

        self.assertAlmostEqual(test_result.pvalue, 0.0)
        self.assertAlmostEqual(test_result.statistic, 1.0)

    def test_linear_function_exhibits_monotonically_correlation(self):
        rng = np.random.default_rng(2)
        a = np.arange(0, 10000)
        b = a * 10
        test_result = extrastats.permutation_test(lambda a, b: sp.stats.kendalltau(a, b)[0],
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  batch=True,
                                                  permutation_type=extrastats.PermutationType.pairings)

        self.assertAlmostEqual(test_result.pvalue, 0.0)
        self.assertAlmostEqual(test_result.statistic, 1.0)

    def test_linear_function_exhibits_inverse_linear_correlation(self):
        rng = np.random.default_rng(2)
        a = np.arange(0, 10000)
        b = a * -10
        test_result = extrastats.permutation_test(lambda a, b: sp.stats.pearsonr(a, b)[0],
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  batch=True,
                                                  permutation_type=extrastats.PermutationType.pairings)

        self.assertAlmostEqual(test_result.pvalue, 0.0)
        self.assertAlmostEqual(test_result.statistic, -1.0)

    def test_exponential_function_exhibits_monotonic_correlation(self):
        rng = np.random.default_rng(2)
        a = np.arange(0, 100)
        b = np.exp(a)
        test_result = extrastats.permutation_test(lambda a, b: sp.stats.kendalltau(a, b)[0],
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  batch=True,
                                                  permutation_type=extrastats.PermutationType.pairings)

        self.assertAlmostEqual(test_result.pvalue, 0.0)
        self.assertAlmostEqual(test_result.statistic, 1.0)

    def test_sinusoidal_function_is_linearly_uncorrelated(self):
        rng = np.random.default_rng(2)
        a = np.arange(0, 10000)
        b = np.sin(a)
        test_result = extrastats.permutation_test(lambda a, b: sp.stats.pearsonr(a, b)[0],
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  batch=True,
                                                  permutation_type=extrastats.PermutationType.pairings)

        self.assertAlmostEqual(test_result.pvalue, 0.996)
        self.assertAlmostEqual(test_result.statistic, 2.67353702e-05)

    def test_sinusoidal_function_is_monotonically_uncorrelated(self):
        rng = np.random.default_rng(2)
        a = np.arange(0, 10000)
        b = np.sin(a)
        test_result = extrastats.permutation_test(lambda a, b: sp.stats.kendalltau(a, b)[0],
                                                  a,
                                                  b,
                                                  random_state=rng,
                                                  batch=True,
                                                  permutation_type=extrastats.PermutationType.pairings)

        self.assertAlmostEqual(test_result.pvalue, 1.0)
        self.assertAlmostEqual(test_result.statistic, 2.2362236e-05)

    def test_paired_means_are_equal1(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        treatment = rng.normal(0, 1, 10000)
        b = a + treatment
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  n_jobs=-1,
                                                  permutation_type=extrastats.PermutationType.samples)

        self.assertAlmostEqual(test_result.pvalue, 0.666)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 49.94551775)

    def test_paired_means_are_equal2(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(0, 1000, 10000)
        treatment = rng.normal(0, 5, 10000)
        b = a + treatment
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  n_jobs=-1,
                                                  permutation_type=extrastats.PermutationType.samples)

        self.assertAlmostEqual(test_result.pvalue, 0.205)
        self.assertAlmostEqual(test_result.statistic[0], 502.044169231)
        self.assertAlmostEqual(test_result.statistic[1], 501.98310374)

    def test_paired_means_are_equal3(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-500, 500, 10000)
        treatment = rng.normal(0, 10, 10000)
        b = a + treatment
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  n_jobs=-1,
                                                  permutation_type=extrastats.PermutationType.samples)

        self.assertAlmostEqual(test_result.pvalue, 0.854)
        self.assertAlmostEqual(test_result.statistic[0], 0.01720119)
        self.assertAlmostEqual(test_result.statistic[1], 0.03449550)

    def test_paired_means_are_not_equal(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-500, 500, 10000)
        treatment = rng.normal(5, 10, 10000)
        b = a + treatment
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  n_jobs=-1,
                                                  permutation_type=extrastats.PermutationType.samples)

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 0.01720119)
        self.assertAlmostEqual(test_result.statistic[1], 5.0344955)

    def test_paired_mean_is_less(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-500, 500, 10000)
        treatment = rng.normal(5, 10, 10000)
        b = a + treatment
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  n_jobs=-1,
                                                  alternative=extrastats.Alternative.less,
                                                  permutation_type=extrastats.PermutationType.samples)

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 0.01720119)
        self.assertAlmostEqual(test_result.statistic[1], 5.0344955)

    def test_paired_mean_is_not_greater(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-500, 500, 10000)
        treatment = rng.normal(5, 10, 10000)
        b = a + treatment
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  n_jobs=-1,
                                                  alternative=extrastats.Alternative.greater,
                                                  permutation_type=extrastats.PermutationType.samples)

        self.assertAlmostEqual(test_result.pvalue, 1.0)
        self.assertAlmostEqual(test_result.statistic[0], 0.01720119)
        self.assertAlmostEqual(test_result.statistic[1], 5.0344955)

    def test_paired_variance_is_not_less(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-500, 500, 10000)
        treatment = rng.normal(5, 10, 10000)
        b = a + treatment
        test_result = extrastats.permutation_test(np.var, a, b,
                                                  random_state=rng,
                                                  n_jobs=-1,
                                                  alternative=extrastats.Alternative.less,
                                                  permutation_type=extrastats.PermutationType.samples)

        self.assertAlmostEqual(test_result.pvalue, 0.133)
        self.assertAlmostEqual(test_result.statistic[0], 84750.7259122)
        self.assertAlmostEqual(test_result.statistic[1], 84826.54957818)

    def test_paired_variance_is_greater(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-500, 500, 10000)
        b = np.where(a > 0, a - 0.1, a + 0.1)
        test_result = extrastats.permutation_test(np.var, a, b,
                                                  random_state=rng,
                                                  n_jobs=-1,
                                                  alternative=extrastats.Alternative.greater,
                                                  permutation_type=extrastats.PermutationType.samples)

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 84750.7259122)
        self.assertAlmostEqual(test_result.statistic[1], 84700.17749948)


class TestIQR(unittest.TestCase):
    """
    Test cases for extrastats.iqr

    """

    def test_s4(self):
        a = np.array([3, 1, 2, 4])
        iqr = extrastats.iqr(a)
        self.assertEqual(iqr, 2)

    def test_s5(self):
        a = np.array([3, 5, 4, 2, 1])
        iqr = extrastats.iqr(a)
        self.assertEqual(iqr, 3)

    def test_s6(self):
        a = np.array([5, 6, 2, 3, 1, 4])
        iqr = extrastats.iqr(a)
        self.assertEqual(iqr, 3)

    def test_s7(self):
        a = np.array([4, 3, 1, 7, 5, 6, 2])
        iqr = extrastats.iqr(a)
        self.assertEqual(iqr, 4)

    def test_large_array(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100000, 100000000)
        iqr = extrastats.iqr(a)
        self.assertAlmostEqual(iqr, 49998.53616906)


class TestTailWeight(unittest.TestCase):
    """
    Tests for extratest.tail_weight

    """

    def test_standard_normal_left(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 1000)
        tw = extrastats.tail_weight(a, side=extrastats.DistSide.left)
        self.assertAlmostEqual(tw, 0.21362931)

    def test_standard_normal_right(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 1000)
        tw = extrastats.tail_weight(a, side=extrastats.DistSide.right)
        self.assertAlmostEqual(tw, 0.18177707)

    def test_standard_normal_both(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 1000)
        tw = extrastats.tail_weight(a, side=extrastats.DistSide.both)
        self.assertAlmostEqual(tw, 0.19770319)

    def test_left_skewed_dist_left(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 10, 1000)
        b = rng.uniform(40, 60, 200)
        a = np.concatenate([b, a])
        tw = extrastats.tail_weight(a, side=extrastats.DistSide.left)
        self.assertAlmostEqual(tw, 0.67276829)

    def test_left_skewed_dist_right(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 10, 1000)
        b = rng.uniform(40, 60, 200)
        a = np.concatenate([b, a])
        tw = extrastats.tail_weight(a, side=extrastats.DistSide.right)
        self.assertAlmostEqual(tw, 0.18488636)

    def test_left_skewed_dist_both(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 10, 1000)
        b = rng.uniform(40, 60, 200)
        a = np.concatenate([b, a])
        tw = extrastats.tail_weight(a, side=extrastats.DistSide.both)
        self.assertAlmostEqual(tw, 0.42882732)

    def test_bimodal_left1(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 5, 1000)
        b = rng.normal(100, 10, 2011)
        a = np.concatenate([a, b])
        tw = extrastats.tail_weight(a, side=extrastats.DistSide.left)
        self.assertAlmostEqual(tw, -0.645384209)

    def test_bimodal_right1(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 5, 1000)
        b = rng.normal(100, 10, 2011)
        a = np.concatenate([a, b])
        tw = extrastats.tail_weight(a, side=extrastats.DistSide.right)
        self.assertAlmostEqual(tw, 0.11993853)

    def test_bimodal_both1(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 5, 1000)
        b = rng.normal(100, 10, 2011)
        a = np.concatenate([a, b])
        tw = extrastats.tail_weight(a, side=extrastats.DistSide.both)
        self.assertAlmostEqual(tw, -0.26272283)

    def test_bimodal_left2(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 20, 1000)
        b = rng.normal(1000, 20, 100)
        a = np.concatenate([a, b])
        tw = extrastats.tail_weight(a, side=extrastats.DistSide.left)
        self.assertAlmostEqual(tw, 0.17367395)

    def test_bimodal_right2(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 20, 1000)
        b = rng.normal(1000, 20, 100)
        a = np.concatenate([a, b])
        tw = extrastats.tail_weight(a, side=extrastats.DistSide.right)
        self.assertAlmostEqual(tw, 0.49642536)

    def test_bimodal_both2(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 20, 1000)
        b = rng.normal(1000, 20, 100)
        a = np.concatenate([a, b])
        tw = extrastats.tail_weight(a, side=extrastats.DistSide.both)
        self.assertAlmostEqual(tw, 0.33504965)


class TestAcceptsRandomState(unittest.TestCase):
    """
    Tests for extrastats.accepts_random_state

    """

    def test_has_attr(self):
        @extrastats.accepts_random_state
        def test():
            pass

        self.assertTrue(hasattr(test, '_accepts_random_state'))


class TestTestTailWeight(unittest.TestCase):
    """
    Test cases for extrastats.test_tail_weight

    """

    def test_standard_normal_vs_long_tailed_less(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 10, 2000)
        b = rng.uniform(100, 2000, 300)
        c = rng.uniform(-2000, 100, 300)
        b = np.concatenate([a, b, c])
        result = extrastats.test_tail_weight(a, b,
                                             alternative=extrastats.Alternative.less,
                                             random_state=rng)

        self.assertAlmostEqual(result.pvalue, 0)
        self.assertAlmostEqual(result.statistic[0], 0.19348302)
        self.assertAlmostEqual(result.statistic[1], 0.67049582)


if __name__ == '__main__':
    unittest.main()
