"""
Test cases for extrastats.py

Copyright 2022 Jerrad Michael Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import json
import lzma
import unittest
from functools import lru_cache
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import scipy as sp
from statsmodels.stats.weightstats import zconfint

import extrastats as es

TEST_DATA = Path("tests/data")


@lru_cache
def load_data(dataset):
    dataset = TEST_DATA / dataset
    with lzma.open(f"{dataset}.json.xz") as fp:
        census_encoded = fp.read()

    return json.loads(census_encoded.decode("utf-8"))


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
        test_result = es.permutation_test(np.mean, a, b, random_state=rng, alternative="lesser")

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 55.58082181)

    def test_uniform_random_mean_is_greater(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(5, 105, 10000)
        test_result = es.permutation_test(np.mean, a, b, random_state=rng, alternative="greater")

        self.assertAlmostEqual(test_result.pvalue, 1)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 55.58082181)

    def test_uniform_random_variance_is_equal(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(100, 200, 10000)
        test_result = es.permutation_test(np.var, a, b, random_state=rng, alternative="two-sided")

        self.assertAlmostEqual(test_result.pvalue, 0.718)
        self.assertAlmostEqual(test_result.statistic[0], 835.50152489)
        self.assertAlmostEqual(test_result.statistic[1], 819.54902091)

    def test_uniform_random_variance_is_not_equal(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 200, 10000)
        test_result = es.permutation_test(np.var, a, b, random_state=rng, alternative="two-sided")

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 835.50152489)
        self.assertAlmostEqual(test_result.statistic[1], 3278.19608364)

    def test_uniform_random_variance_is_not_less(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(100, 200, 10000)
        test_result = es.permutation_test(np.var, a, b, random_state=rng, alternative="lesser")

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
        test_result = es.permutation_test(np.mean, a, b, random_state=rng, iterations=500)

        self.assertAlmostEqual(test_result.pvalue, 0.112)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 50.58082181)

    def test_10000_iterations(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = es.permutation_test(np.mean, a, b, random_state=rng, iterations=10000)

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
        test_result = es.permutation_test(np.var, a, b, random_state=rng, alternative="greater")

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
            permutation_type="independent",
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
            alternative="lesser",
            permutation_type="independent",
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
            alternative="greater",
            permutation_type="independent",
        )

        self.assertAlmostEqual(test_result.pvalue, 0.992)
        self.assertAlmostEqual(test_result.statistic[0], 69.00820654696294)
        self.assertAlmostEqual(test_result.statistic[1], 72.06607121240145)

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
            permutation_type="pairings",
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
            permutation_type="pairings",
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
            permutation_type="pairings",
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
            permutation_type="pairings",
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
            permutation_type="pairings",
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
            permutation_type="pairings",
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
            permutation_type="pairings",
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
            permutation_type="pairings",
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
            permutation_type="pairings",
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
            permutation_type="pairings",
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
            permutation_type="pairings",
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
            permutation_type="pairings",
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
            permutation_type="samples",
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
            permutation_type="samples",
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
            permutation_type="samples",
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
            permutation_type="samples",
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
            alternative="lesser",
            permutation_type="samples",
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
            alternative="greater",
            permutation_type="samples",
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
            alternative="lesser",
            permutation_type="samples",
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
            alternative="greater",
            permutation_type="samples",
        )

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic[0], 84750.7259122)
        self.assertAlmostEqual(test_result.statistic[1], 84700.17749948)

    def test_mean_squared_error(self):
        rng = np.random.default_rng(101)
        a = np.arange(100) * 5 + 7
        test_result = es.permutation_test(
            lambda x, y: -1 * np.mean((x - y)**2),
            a,
            a,
            permutation_type="pairings",
            batch=True,
            random_state=rng,
        )

        self.assertAlmostEqual(test_result.pvalue, 0)
        self.assertAlmostEqual(test_result.statistic, 0)


class TestTailWeight(unittest.TestCase):
    """
    Tests for extratest.tail_weight

    """

    def test_standard_normal_left(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 1000)
        tw = es.tail_weight(a, side="left")
        self.assertAlmostEqual(tw, 0.21362931, 1)

    def test_standard_normal_right(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 1000)
        tw = es.tail_weight(a, side="right")
        self.assertAlmostEqual(tw, 0.18177707, 1)

    def test_standard_normal_both(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 1000)
        tw = es.tail_weight(a, side="both")
        self.assertAlmostEqual(tw, 0.19770319, 1)

    def test_left_skewed_dist_left(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 10, 1000)
        b = rng.uniform(40, 60, 200)
        a = np.concatenate([b, a])
        tw = es.tail_weight(a, side="left")
        self.assertAlmostEqual(tw, 0.67276829, 1)

    def test_left_skewed_dist_right(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 10, 1000)
        b = rng.uniform(40, 60, 200)
        a = np.concatenate([b, a])
        tw = es.tail_weight(a, side="right")
        self.assertAlmostEqual(tw, 0.18488636, 1)

    def test_left_skewed_dist_both(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 10, 1000)
        b = rng.uniform(40, 60, 200)
        a = np.concatenate([b, a])
        tw = es.tail_weight(a, side="both")
        self.assertAlmostEqual(tw, 0.42882732, 1)

    def test_bimodal_left1(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 5, 1000)
        b = rng.normal(100, 10, 2011)
        a = np.concatenate([a, b])
        tw = es.tail_weight(a, side="left")
        self.assertAlmostEqual(tw, -0.645384209, 1)

    def test_bimodal_right1(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 5, 1000)
        b = rng.normal(100, 10, 2011)
        a = np.concatenate([a, b])
        tw = es.tail_weight(a, side="right")
        self.assertAlmostEqual(tw, 0.11993853, 1)

    def test_bimodal_both1(self):
        rng = np.random.default_rng(0)
        a = rng.normal(50, 5, 1000)
        b = rng.normal(100, 10, 2011)
        a = np.concatenate([a, b])
        tw = es.tail_weight(a, side="both")
        self.assertAlmostEqual(tw, -0.26272283, 1)

    def test_bimodal_left2(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 20, 1000)
        b = rng.normal(1000, 20, 100)
        a = np.concatenate([a, b])
        tw = es.tail_weight(a, side="left")
        self.assertAlmostEqual(tw, 0.17367395, 1)

    def test_bimodal_right2(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 20, 1000)
        b = rng.normal(1000, 20, 100)
        a = np.concatenate([a, b])
        tw = es.tail_weight(a, side="right")
        self.assertAlmostEqual(tw, 0.49642536, 1)

    def test_bimodal_both2(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 20, 1000)
        b = rng.normal(1000, 20, 100)
        a = np.concatenate([a, b])
        tw = es.tail_weight(a, side="both")
        self.assertAlmostEqual(tw, 0.33504965, 1)


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
        x = np.concatenate([rng.uniform(-100, 100, 1000), np.full(1, 10000), np.full(1, -10000)])

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
        self.assertAlmostEqual(low, -2.10054411, 1)
        self.assertAlmostEqual(high, 4.37154061, 1)
        self.assertEqual(len(y), 31)

    def test_left_tailed_distribution(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(0, 1, 1000), rng.uniform(-5, -1, 100)])

        low, high = es.adjusted_boxplot(x)
        y = x[np.logical_or(x < low, x > high)]
        self.assertAlmostEqual(low, -3.62692345, 1)
        self.assertAlmostEqual(high, 2.48356375, 1)
        self.assertEqual(len(y), 43)

    @patch("extrastats.extrastats.medcouple", MagicMock(return_value=np.nan))
    def test_medcouple_nan_raises_exception(self):
        rng = np.random.default_rng(510030955)
        x = rng.uniform(-1000000, 1000000, 1000)
        with self.assertRaises(es.MedcoupleError):
            es.adjusted_boxplot(x)

    @patch("extrastats.extrastats.medcouple", MagicMock(return_value=np.nan))
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


class TestMutualInfo(unittest.TestCase):
    def test_mutual_info_with_random_integers(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 5, 1000)
        y = rng.integers(0, 5, 1000)
        mi = es.mutual_info(x, y)
        self.assertLess(mi, 0.012)

    def test_mutual_info_with_linear_data(self):
        x = np.repeat(np.arange(10), 10)
        y = x * 3 + 9
        mi = es.mutual_info(x, y)
        self.assertAlmostEqual(mi, 1)

    def test_mutual_info_with_exponential_data(self):
        x = np.repeat(np.arange(10), 10)
        y = x**x
        mi = es.mutual_info(x, y)
        self.assertAlmostEqual(mi, 1)

    def test_mutual_info_with_sinusoidal_data(self):
        x = np.repeat(np.arange(10), 10)
        y = np.sin(x)
        _, y_binned = np.unique(y, return_inverse=True)
        mi = es.mutual_info(x, y_binned)
        self.assertAlmostEqual(mi, 1)

    def test_mutual_info_with_50_perc_random_data(self):
        x = np.repeat(np.arange(10), 10)
        y = x * 10
        selector = np.where(np.arange(y.size) % 2 == 0)[0]
        rng = np.random.default_rng(0)
        np.put(y, selector, rng.choice(np.unique(y), selector.size))
        mi = es.mutual_info(x, y)
        self.assertAlmostEqual(mi, 0.45, 2)  # Target value calculated manually

    def test_mutual_info_with_25_perc_random_data(self):
        x = np.repeat(np.arange(10), 10)
        y = x * 10
        selector = np.where(np.arange(y.size) % 4 == 0)[0]
        rng = np.random.default_rng(0)
        np.put(y, selector, rng.choice(np.unique(y), selector.size))
        mi = es.mutual_info(x, y)
        self.assertAlmostEqual(mi, 0.7, 2)  # Target value calculated manually

    def test_mutual_info_with_non_uniform_data(self):
        x = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        x = np.repeat(x, 10)
        y = x * 5
        mi = es.mutual_info(x, y)
        self.assertAlmostEqual(mi, 1)

    def test_mutual_info_with_non_uniform_50_perc_random_data(self):
        x = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        x = np.repeat(x, 10)
        y = x * 5
        selector = np.where(np.arange(y.size) % 2 == 0)[0]
        rng = np.random.default_rng(0)
        np.put(y, selector, rng.choice(np.unique(y), selector.size))
        mi = es.mutual_info(x, y)
        self.assertAlmostEqual(mi, 0.25, 2)

    def test_mutual_info_with_different_size_arrays(self):
        x = np.arange(10)
        y = np.arange(9)
        with self.assertRaises(ValueError):
            es.mutual_info(x, y)

    def test_mutual_info_with_empty_arrays(self):
        mi = es.mutual_info([], [])
        self.assertEqual(mi, 0)

    def test_mutual_info_with_size1_arrays(self):
        mi = es.mutual_info([1], [3])
        self.assertEqual(mi, 0)

    def test_mutual_info_with_size2_arrays(self):
        mi = es.mutual_info([1, 2], [3, 4])
        self.assertEqual(mi, 1)

    def test_mutual_info_with_one_unique_label(self):
        x = np.ones(100)
        y = np.zeros(100)
        mi = es.mutual_info(x, y)
        self.assertEqual(mi, 0)

    def test_mutual_info_with_max_info_norm(self):
        x = np.repeat(np.arange(10), 10)
        y = x * 10
        selector = np.where(np.arange(y.size) % 2 == 0)[0]
        np.put(y, selector, -1)
        mi = es.mutual_info(x, y, norm="max")
        self.assertAlmostEqual(mi, 0.5, 2)

    def test_mutual_info_with_avg_info_norm(self):
        x = np.repeat(np.arange(10), 10)
        y = x * 10
        selector = np.where(np.arange(y.size) % 2 == 0)[0]
        np.put(y, selector, -1)
        mi = es.mutual_info(x, y, norm="avg")
        self.assertAlmostEqual(mi, 0.56, 2)

    def test_mutual_info_with_invalid_norm(self):
        with self.assertRaises(ValueError):
            es.mutual_info([], [], norm=None)

    def test_mutual_info_with_base_e(self):
        x = np.repeat(np.arange(10), 10)
        y = x * 10
        selector = np.where(np.arange(y.size) % 2 == 0)[0]
        np.put(y, selector, -1)
        mi = es.mutual_info(x, y, norm="none", base=np.e)
        self.assertAlmostEqual(mi, 1.15, 2)

    def test_mutual_info_with_base_0(self):
        with self.assertRaises(ValueError):
            es.mutual_info([], [], base=0)

    def test_mutual_info_with_negative_base(self):
        with self.assertRaises(ValueError):
            es.mutual_info([], [], base=-1)

    def test_mutual_info_with_titanic_data(self):
        titanic = load_data("titanic")
        mi = es.mutual_info(titanic["sex"], titanic["survived"])
        self.assertAlmostEqual(mi, 0.23, 2)

    def test_mutual_info_with_census_occupation_education(self):
        census = load_data("census")
        mi = es.mutual_info(census["occupation"], census["education"])
        self.assertAlmostEqual(mi, 0.11, 2)

    def test_mutual_info_with_census_marital_status_relationship(self):
        census = load_data("census")
        mi = es.mutual_info(census["marital_status"], census["relationship"])
        self.assertAlmostEqual(mi, 0.57, 2)


class TestConfidenceInterval(unittest.TestCase):
    """Test cases for extrastats.confidence_interval"""

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def assert_confidence_intervals(self, confidence_intervals, expected_intervals, precision=0.1):
        """
        Assert that 'confidence_intervals' agree with 'expected_intervals' within
        a range indicated by 'precision'.

        """

        for level, (expected_lower, expected_upper) in expected_intervals.items():
            self.assertLessEqual(abs(expected_lower - confidence_intervals[level][0]), precision)
            self.assertLessEqual(abs(expected_upper - confidence_intervals[level][1]), precision)

    def mean_test(self, n, precision=0.1, levels=None, **kwargs):
        """
        Test confidence_interval with the arithmetic mean and a random
        sample from a standard normal distribution, with the sample size
        (n) specified by individual test cases.

        """

        sample = self.rng.normal(0, 1, n)
        if levels:
            kwargs["levels"] = levels

        else:
            levels = 0.9, 0.95, 0.99

        confidence_intervals, bootstrap_data = es.confidence_interval(
            np.mean, sample, random_state=self.rng, **kwargs
        )

        self.assertEqual(len(confidence_intervals), len(levels))
        mean = np.mean(sample)
        std_err = sp.stats.sem(sample)
        expected_intervals = {}
        for level in levels:
            # Compute margin of error
            h = std_err * sp.stats.t.ppf((1 + level) / 2, n - 1)
            lower_bound = mean - h
            upper_bound = mean + h
            expected_intervals[level] = lower_bound, upper_bound

        self.assert_confidence_intervals(confidence_intervals, expected_intervals, precision)
        return confidence_intervals, bootstrap_data

    @unittest.expectedFailure
    def test_mean_with_small_n(self):
        """
        Test mean statistic with n = 20.

        Currently, this is expected to fail because the implementation wasn't
        designed for small sizes with high confidence levels.

        """

        n = 20
        self.mean_test(n)

    def test_mean_with_moderate_n(self):
        """
        Test mean statistic with n = 50.

        """

        n = 50
        self.mean_test(n)

    def test_mean_with_large_n(self):
        """
        Test mean statistic with n = 500.

        """

        n = 500
        self.mean_test(n, precision=2)

    def test_mean_difference_with_moderate_n(self):
        """
        Test group mean difference statistic with n = 50.

        """

        n = 50
        a = self.rng.normal(0, 1, n)
        b = self.rng.normal(1, 1, n)

        def mean_diff(a, b):
            return np.mean(a) - np.mean(b)

        confidence_intervals, _ = es.confidence_interval(mean_diff, a, b, random_state=self.rng)
        confidence_levels = 0.9, 0.95, 0.99
        expected_intervals = {}
        for level in confidence_levels:
            lower, upper = zconfint(a, b, alpha=1 - level)
            expected_intervals[level] = lower, upper

        self.assert_confidence_intervals(confidence_intervals, expected_intervals)

    def test_mean_difference_of_three_groups(self):
        """
        Test mean difference of three groups.
        """

        n = 50
        a = self.rng.normal(0, 1, n)
        b = self.rng.normal(0, 1, n)
        c = self.rng.normal(1, 1, n)

        def grand_mean_diff(a, b, c):
            groups = np.array([a, b, c])
            grand_mean = np.mean(groups)
            return np.mean(c) - grand_mean

        confidence_intervals, _ = es.confidence_interval(
            grand_mean_diff,
            a,
            b,
            c,
            random_state=self.rng,
        )

        # Compute grand mean and standard error for group c
        groups = np.array([a, b, c])
        grand_mean = np.mean(groups)
        c_mean_diff = np.mean(c) - grand_mean
        c_std_err = sp.stats.sem(c)

        # Correct degrees of freedom
        df = 3 * n - 3

        confidence_levels = 0.9, 0.95, 0.99
        expected_intervals = {}
        for level in confidence_levels:
            alpha = 1 - level
            t_crit = sp.stats.t.ppf(1 - alpha / 2, df)
            lower = c_mean_diff - t_crit * c_std_err
            upper = c_mean_diff + t_crit * c_std_err
            expected_intervals[level] = lower, upper

        self.assert_confidence_intervals(confidence_intervals, expected_intervals)

    def test_default_iterations(self):
        """
        Test with default number of iterations.

        """

        n = 50
        default_iterations = 2000
        _, bootstrap_data = self.mean_test(n)
        self.assertEqual(len(bootstrap_data), default_iterations)
        self.assertEqual(bootstrap_data.dtype, np.float64)

    def test_non_default_iterations(self):
        """
        Test with non-default number of iterations.

        """

        n = 50
        iterations = 3000
        _, bootstrap_data = self.mean_test(n, iterations=iterations)
        self.assertEqual(len(bootstrap_data), iterations)
        self.assertEqual(bootstrap_data.dtype, np.float64)

    def test_non_default_levels(self):
        """
        Test with non-default confidence levels.

        """

        n = 50
        confidence_levels = (0.8,)
        self.mean_test(n, levels=confidence_levels)

    def test_non_default_n_jobs(self):
        """
        Test with a non-default number value for n_jobs.

        """

        n = 50
        n_jobs = 2
        self.mean_test(n, n_jobs=n_jobs)

    def test_non_default_parallel(self):
        """
        Test with a non-default value for parallel.

        """

        n = 50
        parallel = joblib.Parallel()
        self.mean_test(n, parallel=parallel)

    def test_default_random_state(self):
        """
        Test with the default value for random_state.

        """

        n = 500
        a = self.rng.normal(0, 1, n)
        b = self.rng.normal(1, 1, n)

        def mean_diff(a, b):
            return np.mean(a) - np.mean(b)

        confidence_intervals, _ = es.confidence_interval(mean_diff, a, b)
        confidence_levels = (0.9, 0.95)
        expected_intervals = {}
        for level in confidence_levels:
            expected_intervals[level] = zconfint(a, b, alpha=1 - level)

        self.assert_confidence_intervals(confidence_intervals, expected_intervals)

    def test_random_state_integer(self):
        """
        Test with an integer value for random_state.

        """

        n = 50
        a = self.rng.normal(0, 1, n)
        b = self.rng.normal(1, 1, n)

        def mean_diff(a, b):
            return np.mean(a) - np.mean(b)

        confidence_intervals, _ = es.confidence_interval(mean_diff, a, b, random_state=0)
        confidence_levels = (0.9, 0.95, 0.99)
        expected_intervals = {}
        for level in confidence_levels:
            expected_intervals[level] = zconfint(a, b, alpha=1 - level)

        self.assert_confidence_intervals(confidence_intervals, expected_intervals)

    def test_a_is_list(self):
        """
        Test with lists for a and args parameters.

        """

        n = 50
        a = list(self.rng.normal(0, 1, n))
        b = list(self.rng.normal(1, 1, n))

        def mean_diff(a, b):
            return np.mean(a) - np.mean(b)

        confidence_intervals, _ = es.confidence_interval(mean_diff, a, b, random_state=self.rng)
        confidence_levels = (0.9, 0.95, 0.99)
        expected_intervals = {}
        for level in confidence_levels:
            expected_intervals[level] = zconfint(a, b, alpha=1 - level)

        self.assert_confidence_intervals(confidence_intervals, expected_intervals)

    def test_random_state_raises_value_error(self):
        """
        Test that an unexpected value for random_state raises a ValueError.

        """

        n = 50
        sample = self.rng.normal(0, 1, n)
        with self.assertRaises(ValueError):
            es.confidence_interval(np.mean, sample, random_state="duck")

    def test_warns_on_unequal_sample_sizes(self):
        """
        Test that a warning is raised when groups are different sizes.

        """

        a = self.rng.normal(0, 1, 20)
        b = self.rng.normal(0, 1, 20)
        c = self.rng.normal(0, 1, 30)

        with self.assertWarns(UserWarning):
            es.confidence_interval(lambda *_: 0, a, b, c)

    def test_with_n_eq_0(self):
        """
        Test that a ValueError is raised when n=0.

        """

        with self.assertRaises(ValueError):
            es.confidence_interval(lambda _: 0, [])

    def test_with_n_eq_1(self):
        """
        Test that a ValueError is raised when n=1.

        """

        with self.assertRaises(ValueError):
            es.confidence_interval(lambda _: 0, [0])

    def test_iterations_lt_1(self):
        """
        Test that a ValueError is raised when iterations < 1.

        """

        with self.assertRaises(ValueError):
            es.confidence_interval(lambda _: 0, [], iterations=0)

    def test_zero_levels(self):
        """
        Test that a ValueError is raised when there are no confidence levels.

        """

        with self.assertRaises(ValueError):
            es.confidence_interval(lambda _: 0, [], levels=tuple())

    def test_invalid_levels(self):
        """
        Test that a ValueError is raised on invalid confidence levels.

        """

        with self.assertRaises(ValueError):
            es.confidence_interval(lambda _: 0, [], levels=(0.9, -5))


class TestSampleSize(unittest.TestCase):
    """Test cases for extrastats.sample_size"""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.dummy_model = lambda n, rng: rng.normal(size=n)
        self.dummy_statistic = np.mean

    def test_sample_size_success(self):
        estimated_sample_size = es.sample_size(
            lambda n, rng: [rng.normal(100, 10, n)],
            np.mean,
            1,
            lower=1000,
            upper=2000,
            iterations=2000,
            mc_iterations=500,
            n_jobs=-1,
            random_state=self.rng,
        )

        self.assertTrue(1500 <= estimated_sample_size <= 1700)

    def test_empirical_model_success(self):
        x = self.rng.normal(100, 10, 20)
        estimated_sample_size = es.sample_size(
            x,
            np.mean,
            1,
            lower=1000,
            upper=1500,
            iterations=2000,
            mc_iterations=500,
            n_jobs=-1,
            random_state=self.rng,
        )

        self.assertTrue(1100 <= estimated_sample_size <= 1200)

    def test_sample_size_for_mean_failure(self):
        x = self.rng.normal(100, 10, 20)

        with self.assertRaises(RuntimeError):
            es.sample_size(
                x,
                np.mean,
                1,
                lower=50,
                upper=100,
                iterations=500,
                mc_iterations=500,
                n_jobs=-1,
                random_state=self.rng,
            )

    def test_lower_bound_validation(self):
        with self.assertRaises(ValueError) as context:
            es.sample_size(self.dummy_model, self.dummy_statistic, width=0.5, lower=1)

        self.assertIn("Argument for 'lower' must be >= 2", str(context.exception))

    def test_upper_bound_validation(self):
        with self.assertRaises(ValueError) as context:
            es.sample_size(self.dummy_model, self.dummy_statistic, width=0.5, lower=10, upper=5)

        self.assertIn("Argument for 'upper' must be > 'lower'", str(context.exception))

    def test_width_validation(self):
        with self.assertRaises(ValueError) as context:
            es.sample_size(self.dummy_model, self.dummy_statistic, width=-0.5)

        self.assertIn("Argument for 'width' must be >= 0", str(context.exception))

    def test_prob_validation(self):
        with self.assertRaises(ValueError) as context:
            es.sample_size(self.dummy_model, self.dummy_statistic, width=0.5, prob=1.1)

        self.assertIn("Argument for 'prob' must be within range [0, 1]", str(context.exception))

        with self.assertRaises(ValueError) as context:
            es.sample_size(self.dummy_model, self.dummy_statistic, width=0.5, prob=-0.1)

        self.assertIn("Argument for 'prob' must be within range [0, 1]", str(context.exception))

    def test_level_validation(self):
        with self.assertRaises(ValueError) as context:
            es.sample_size(self.dummy_model, self.dummy_statistic, width=0.5, level=1.1)

        self.assertIn("Argument for 'level' must be within range (0, 1)", str(context.exception))

        with self.assertRaises(ValueError) as context:
            es.sample_size(self.dummy_model, self.dummy_statistic, width=0.5, level=0.0)

        self.assertIn("Argument for 'level' must be within range (0, 1)", str(context.exception))

    def test_mc_iterations_validation(self):
        with self.assertRaises(ValueError) as context:
            es.sample_size(self.dummy_model, self.dummy_statistic, width=0.5, mc_iterations=0)

        self.assertIn("Argument for 'mc_iterations' must be >= 1", str(context.exception))

    def test_convergence_limit_validation(self):
        with self.assertRaises(ValueError) as context:
            es.sample_size(self.dummy_model, self.dummy_statistic, width=0.5, convergence_limit=-1)

        self.assertIn("Argument for 'convergence_limit' must be >= 0", str(context.exception))

    def test_tol_validation(self):
        with self.assertRaises(ValueError) as context:
            es.sample_size(self.dummy_model, self.dummy_statistic, width=0.5, tol=-0.1)

        self.assertIn("Argument for 'tol' must be >= 0", str(context.exception))

    def test_n_jobs_validation(self):
        with self.assertRaises(ValueError) as context:
            es.sample_size(self.dummy_model, self.dummy_statistic, width=0.5, n_jobs=-2)

        self.assertIn("Argument for 'n_jobs' must be >= -1", str(context.exception))

    def test_random_state_validation(self):
        with self.assertRaises(ValueError) as context:
            es.sample_size(
                self.dummy_model, self.dummy_statistic, width=0.5, random_state="invalid"
            )

        self.assertIn("Got unexpected value for random_state", str(context.exception))


if __name__ == "__main__":
    unittest.main()
