"""
Test cases for extrastats.py

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

    def test_uniform_random_with_two_cpus(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  n_jobs=2)

        self.assertAlmostEqual(test_result.pvalue, 0.113)
        self.assertAlmostEqual(test_result.statistic[0], 49.94106601)
        self.assertAlmostEqual(test_result.statistic[1], 50.58082181)

    def test_uniform_random_with_all_cpus(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, 10000)
        b = rng.uniform(0, 100, 10000)
        test_result = extrastats.permutation_test(np.mean, a, b,
                                                  random_state=rng,
                                                  n_jobs=-1)

        self.assertAlmostEqual(test_result.pvalue, 0.113)
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
        self.assertAlmostEqual(test_result.statistic[0], 81.41777213378225)
        self.assertAlmostEqual(test_result.statistic[1], 84.51104547699504)

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
        self.assertAlmostEqual(test_result.statistic[0], 89.55954934716047)
        self.assertAlmostEqual(test_result.statistic[1], 84.51104547699504)


if __name__ == '__main__':
    unittest.main()
