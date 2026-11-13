import unittest

import numpy as np

from xai import stats


class StatsTests(unittest.TestCase):
    '''Unit tests for xai.stats (corrected dose-response analysis)'''

    def test_chance_correction(self):
        '''Pet dose -0.5: raw 17.5, accuracy without shortcut 82.4 gives floor 17.6'''
        floor = stats.chance_floor(82.4, 2)
        self.assertAlmostEqual(float(floor), 17.6)
        self.assertAlmostEqual(float(stats.corrected_reliance(17.6, floor)), 0.0)
        self.assertAlmostEqual(float(stats.chance_floor(91.0, 10)), 1.0)
        self.assertAlmostEqual(float(stats.corrected_reliance(100, 5)), 100.0)

    def test_win_scores_and_gap(self):
        '''deletion: lower AUC wins; ties count as half'''
        w = stats.win_scores([0.1, 0.5, 0.3], [0.2, 0.4, 0.3], higher=False, tol=1e-4)
        np.testing.assert_array_equal(w, [1, 0, 0.5])
        np.testing.assert_allclose(stats.auc_gap([0.1, 0.5], [0.2, 0.4], higher=False), [0.1, -0.1])
        np.testing.assert_allclose(stats.auc_gap([0.1, 0.5], [0.2, 0.4], higher=True), [-0.1, 0.1])

    def test_pava(self):
        '''PAVA on a hand example'''
        np.testing.assert_allclose(stats.pava([1, 3, 2, 4, 3, 5]), [1, 2.5, 2.5, 3.5, 3.5, 5])
        np.testing.assert_allclose(stats.pava([3, 2, 1]), [2, 2, 2])

    def test_isotonic_switch(self):
        '''crossing between x = 20 (0.4) and x = 40 (0.6) is at 30'''
        switch, status = stats.isotonic_switch([0, 20, 40, 60], [0.1, 0.4, 0.6, 0.9])
        self.assertEqual(status, "crossed")
        self.assertAlmostEqual(switch, 30.0)
        self.assertEqual(stats.isotonic_switch([0, 10], [0.6, 0.7])[1], "above 0.5 at lowest reliance")
        self.assertEqual(stats.isotonic_switch([0, 10], [0.1, 0.2])[1], "no crossing")

    def test_identification(self):
        '''a switch at 34 with models at 23 and 45 is not identified'''
        distance, status = stats.switch_identification(34.0, [0, 10, 23, 45, 90])
        self.assertAlmostEqual(distance, 11.0)
        self.assertTrue(status.startswith("not identified"))
        self.assertEqual(stats.switch_identification(30.0, [23, 45])[1], "identified")

    def test_logistic4(self):
        '''4PL recovers a synthetic switch within 1 point'''
        x = np.linspace(0, 100, 41)
        y = 0.2 + (0.9 - 0.2) / (1 + np.exp(-(x - 40) / 6))
        truth = 40 - 6 * np.log((0.9 - 0.2) / (0.5 - 0.2) - 1)
        switch, params, ok = stats.logistic4_switch(x, y)
        self.assertTrue(ok)
        self.assertLess(abs(switch - truth), 1.0)

    def test_bh_holm(self):
        '''Holm on [0.01, 0.04, 0.03] gives [0.03, 0.06, 0.06]; BH textbook vector'''
        np.testing.assert_allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(stats.bh([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])
        out = stats.holm([0.01, np.nan, 0.03])
        self.assertTrue(np.isnan(out[1]))
        np.testing.assert_allclose(out[[0, 2]], [0.02, 0.03])

    def test_format_p(self):
        self.assertEqual(stats.format_p(0.0002), "< 0.001")
        self.assertEqual(stats.format_p(0.0123), "0.012")

    def test_auroc(self):
        '''AUROC 1, 0.5 (ties) and 0; NaN with one class'''
        labels = [0, 0, 1, 1]
        self.assertEqual(stats.auroc([0.1, 0.2, 0.8, 0.9], labels), 1.0)
        self.assertEqual(stats.auroc([0.5, 0.5, 0.5, 0.5], labels), 0.5)
        self.assertEqual(stats.auroc([0.9, 0.8, 0.2, 0.1], labels), 0.0)
        self.assertTrue(np.isnan(stats.auroc([0.1, 0.2], [1, 1])))
        rows = stats.auroc_rows([[0.1, 0.2, 0.8, 0.9], [0.9, 0.8, 0.2, 0.1], [1, 2, 3, 4]],
                                [labels, labels, [1, 1, 1, 1]])
        np.testing.assert_array_equal(rows[:2], [1.0, 0.0])
        self.assertTrue(np.isnan(rows[2]))

    def test_decision_rates(self):
        '''CIFAR-like: 5 models below 50% all say shortcut, 1 above'''
        rates = stats.decision_rates([0.6] * 6, [5, 10, 20, 30, 40, 90])
        self.assertEqual(rates["balanced_accuracy"], 0.5)
        self.assertEqual(rates["fpr"], 1.0)
        self.assertEqual(rates["tpr"], 1.0)
        self.assertEqual((rates["n_above"], rates["n_below"]), (1, 5))

    def test_kendall_null(self):
        '''n = 5: 120 orderings; P(T >= 0.2) = 49/120, P(T >= 0.6) = 14/120, P(T >= 0.8) = 5/120'''
        null = stats.kendall_null(5)
        self.assertEqual(int(null["count"].sum()), 120)
        self.assertEqual(list(null["count"]), [1, 4, 9, 15, 20, 22, 20, 15, 9, 4, 1])
        self.assertAlmostEqual(stats.kendall_exact_p(0.2)[0], 49 / 120)
        self.assertAlmostEqual(stats.kendall_exact_p(0.6)[0], 14 / 120)
        self.assertAlmostEqual(stats.kendall_exact_p(0.8)[0], 5 / 120)
        self.assertAlmostEqual(stats.kendall_exact_p(0.0)[1], 1.0)
        mean = stats.mean_tau_null(5, 3)
        self.assertAlmostEqual(mean.prob.sum(), 1.0)
        self.assertAlmostEqual(float((mean.tau_mean * mean.prob).sum()), 0.0)

    def test_partial_spearman(self):
        '''x and y both driven by z only: partial correlation near 0'''
        rng = np.random.default_rng(0)
        z = rng.normal(size=200)
        x, y = z + 0.1 * rng.normal(size=200), z + 0.1 * rng.normal(size=200)
        rho, p = stats.partial_spearman(x, y, z)
        self.assertLess(abs(rho), 0.3)
        rho, p = stats.partial_spearman(x, x, z)
        self.assertAlmostEqual(rho, 1.0)

    def test_spearman_rows(self):
        rows = stats.spearman_rows(np.arange(5), np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]))
        np.testing.assert_allclose(rows, [1, -1])

    def test_two_level_picks(self):
        '''every model pick stays inside its dose; image picks are within range'''
        doses = np.array([-0.4, -0.4, -0.4, 0.0, 0.0, 0.1])
        P, I = stats.two_level_picks(doses, 100, 200, 0)
        self.assertEqual(P.shape, (200, 6))
        self.assertEqual(I.shape, (200, 6, 100))
        self.assertTrue(np.all(np.isclose(doses[P], doses[None, :])))
        self.assertTrue((I >= 0).all() and (I < 100).all())
        self.assertTrue(np.all(P[:, 5] == 5))

    def test_crossed_picks(self):
        '''models stay inside their dose; every slot of a replicate shares one image draw'''
        doses = np.array([-0.4, -0.4, -0.4, 0.0, 0.0, 0.1])
        P, I = stats.crossed_picks(doses, 100, 200, 0)
        self.assertEqual(P.shape, (200, 6))
        self.assertEqual(I.shape, (200, 6, 100))
        self.assertTrue(np.all(np.isclose(doses[P], doses[None, :])))
        self.assertTrue((I == I[:, :1]).all())
        self.assertFalse((I[0] == I[1]).all())
        self.assertTrue((I >= 0).all() and (I < 100).all())

    def test_clopper_pearson(self):
        '''k = 0 of 4: low 0; k = 3 of 4: about [0.194, 0.994]; n = 0: nan'''
        low, high = stats.clopper_pearson(0, 4)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.6024, places=3)
        low, high = stats.clopper_pearson(3, 4)
        self.assertAlmostEqual(low, 0.194, places=3)
        self.assertAlmostEqual(high, 0.994, places=3)
        self.assertEqual(stats.clopper_pearson(4, 4)[1], 1.0)
        self.assertTrue(np.isnan(stats.clopper_pearson(0, 0)[0]))

    def test_bca(self):
        '''symmetric normal replicates: BCa close to the percentile interval; constant: nan'''
        rng = np.random.default_rng(0)
        boot = rng.normal(0.0, 1.0, 4000)
        jack = rng.normal(0.0, 0.01, 50)
        low, high = stats.bca_interval(0.0, boot, jack)
        p_low, p_high = np.percentile(boot, [2.5, 97.5])
        self.assertLess(abs(low - p_low), 0.1)
        self.assertLess(abs(high - p_high), 0.1)
        self.assertTrue(np.isnan(stats.bca_interval(0.5, np.full(100, 0.5), jack)[0]))
        self.assertTrue(np.isnan(stats.bca_interval(np.nan, boot, jack)[0]))

    def test_jitter(self):
        '''jitter breaks ties without changing the order of distinct values'''
        x = np.array([1.0, 1.0, 2.0, 3.0])
        out = stats.jitter(x, np.random.default_rng(0))
        self.assertEqual(len(np.unique(out)), 4)
        np.testing.assert_array_equal(np.argsort(out)[2:], [2, 3])
        self.assertLess(np.abs(out - x).max(), 1e-8)


if __name__ == "__main__":
    unittest.main()
