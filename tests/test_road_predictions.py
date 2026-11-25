"""P10 and P11 of docs/PREDICTIONS_sd3_sd4.md (xai/dose_analysis.py road_p10, road_p11)."""
import unittest

import numpy as np
import pandas as pd

from xai import dose_analysis

TITLES = {"deletion_zero": "Deletion (black)", "road_morf": "ROAD MoRF", "road_lerf": "ROAD LeRF"}
IMAGES = 6


def settings_for(values, rc=None, keys=None, reps=50):
    """ROAD-like Settings for the registered P10 settings (lowest dose and dose 0).
    values[(base, shortcut)] = {column suffix: AUC}, e.g. {"deletion_zero_random": 0.2}; every
    other map/metric AUC is 0.5. rc: r_c per (dose, seed) (default 0 at the lowest dose, 90 at 0)."""
    models, oracle = [], []
    for key in keys or dose_analysis.P10_LOWEST:
        dose, seeds = dose_analysis.P10_LOWEST.get(key, (-0.5, (42, 1, 2)))
        for d in (dose, 0.0):
            for s in seeds:
                r = (rc or {}).get((d, s), 0.0 if d == dose else 90.0)
                models.append({"base": key[0], "shortcut": key[1], "dose": d, "seed": s, "reliance": r,
                               "reliance_corrected": r})
                for i in range(IMAGES):
                    row = {"base": key[0], "shortcut": key[1], "dose": d, "seed": s, "index": i, "label": 0}
                    for metric in TITLES:
                        for m in ("shortcut", "object", "random"):
                            row[f"{metric}_{m}"] = 0.5
                    if d == dose:
                        row.update(values.get(key, {}))
                    oracle.append(row)
    models, oracle = pd.DataFrame(models), pd.DataFrame(oracle)
    return dose_analysis.build_settings(oracle, None, models, reps, 0, 1e-4)


class PairedImageBootstrap(unittest.TestCase):
    def test_matches_a_loop_and_is_seeded(self):
        rng = np.random.default_rng(3)
        a, b = rng.random((3, 7)), rng.random((3, 7))
        point, boot = dose_analysis.paired_image_bootstrap(a, b, 20, 0)
        self.assertAlmostEqual(point, float((a.mean(axis=1) - b.mean(axis=1)).mean()))
        picks = np.random.default_rng(0).integers(0, 7, size=(20, 3, 7))
        loop = [np.mean([a[m, picks[r, m]].mean() - b[m, picks[r, m]].mean() for m in range(3)]) for r in range(20)]
        np.testing.assert_allclose(boot, loop)
        np.testing.assert_array_equal(boot, dose_analysis.paired_image_bootstrap(a, b, 20, 0)[1])

    def test_same_images_for_both_metrics(self):
        a = np.arange(12, dtype=float).reshape(2, 6)
        point, boot = dose_analysis.paired_image_bootstrap(a, a - 0.25, 100, 0)
        self.assertAlmostEqual(point, 0.25)
        np.testing.assert_allclose(boot, 0.25)  # identical resamples cancel exactly


class P10(unittest.TestCase):
    def test_holds_when_road_lowers_the_random_preference_everywhere(self):
        # black: random (0.2) beats object (0.5) on every image; ROAD MoRF: random 0.8 loses
        values = {k: {"deletion_zero_random": 0.2, "road_morf_random": 0.8} for k in dose_analysis.P10_LOWEST}
        p10 = dose_analysis.road_p10(settings_for(values), TITLES, 200, 0)
        morf = p10[p10.road_metric == "ROAD MoRF"]
        cells = morf[morf.base != "overall"]
        self.assertEqual(list(cells.verdict), ["holds"] * 4)
        np.testing.assert_allclose(cells.difference, 1.0)
        np.testing.assert_allclose(cells.pref_ro_black, 1.0)
        self.assertEqual(morf[morf.base == "overall"].verdict.iloc[0], "holds")
        self.assertEqual(list(cells.models), [2, 3, 3, 3])

    def test_refuted_in_one_setting_refutes_overall(self):
        values = {k: {"deletion_zero_random": 0.2, "road_morf_random": 0.8} for k in dose_analysis.P10_LOWEST}
        values[("cifar10", "patch")] = {"deletion_zero_random": 0.2, "road_morf_random": 0.2}  # equal: diff 0
        p10 = dose_analysis.road_p10(settings_for(values), TITLES, 200, 0)
        morf = p10[p10.road_metric == "ROAD MoRF"].set_index("base")
        self.assertEqual(morf.loc["cifar10", "verdict"], "refuted")
        self.assertEqual(morf.loc["overall", "verdict"], "refuted")

    def test_floor_and_missing_settings_are_not_testable(self):
        values = {k: {"deletion_zero_random": 0.2, "road_morf_random": 0.8} for k in dose_analysis.P10_LOWEST}
        values[("cifar10", "patch")] = {"deletion_zero_random": 0.8}  # black preference 0 (floor)
        keys = [k for k in dose_analysis.P10_LOWEST if k != ("pet_resnet18_pretrained", "patch")]
        p10 = dose_analysis.road_p10(settings_for(values, keys=keys), TITLES, 200, 0)
        morf = p10[p10.road_metric == "ROAD MoRF"].set_index("base")
        self.assertEqual(morf.loc["cifar10", "verdict"], "not testable")
        self.assertEqual(morf.loc["pet_resnet18_pretrained", "verdict"], "not testable")
        self.assertEqual(morf.loc["overall", "verdict"], "not testable")  # 2 of 4 testable

    def test_positive_difference_with_interval_through_zero_is_not_confirmed(self):
        values = {k: {"deletion_zero_random": 0.2, "road_morf_random": 0.8} for k in dose_analysis.P10_LOWEST}
        pet = ("pet_resnet18", "patch")
        settings = settings_for(values)
        s = next(x for x in settings if (x.base, x.shortcut) == pet)
        # per image, black - ROAD win: +1 on images 1 and 3, -1 on image 2, 0 elsewhere (mean 1/6)
        low = s.oracle.dose == -0.5
        s.oracle.loc[low, "road_morf_random"] = 0.2
        s.oracle.loc[low & s.oracle["index"].isin([1, 3]), "road_morf_random"] = 0.8
        s.oracle.loc[low & (s.oracle["index"] == 2), "deletion_zero_random"] = 0.8
        s._cache.clear()
        row = dose_analysis.road_p10(settings, TITLES, 500, 0).set_index(["road_metric", "base"]).loc[
            ("ROAD MoRF", "pet_resnet18")]
        self.assertGreater(row.difference, 0)
        self.assertLessEqual(row.ci_low, 0)
        self.assertEqual(row.verdict, "not confirmed")


class P11(unittest.TestCase):
    def test_rates_discordance_and_verdict(self):
        pet = ("pet_resnet18", "patch")
        # lowest-dose models (r_c 0): black says shortcut (0.2 < 0.5), ROAD MoRF does not
        rc = {(-0.5, 42): 0.0, (-0.5, 1): 0.0, (-0.5, 2): 60.0}
        values = {pet: {"deletion_zero_shortcut": 0.2, "road_morf_shortcut": 0.8}}
        p11 = dose_analysis.road_p11(settings_for(values, rc=rc, keys=[pet]), TITLES)
        row = p11[(p11.base == "pet_resnet18") & (p11.road_metric == "ROAD MoRF")].iloc[0]
        self.assertEqual(row.role, "primary")
        self.assertEqual((row.n_below, row.k_black, row.k_road), (2, 2, 0))
        self.assertAlmostEqual(row.fpr_black, 1.0)
        self.assertAlmostEqual(row.difference, 1.0)
        self.assertEqual((row.discordant_black_only, row.discordant_road_only), (2, 0))
        self.assertAlmostEqual(row.p_sign_two_sided, 0.5)
        self.assertEqual(row.verdict, "holds")
        self.assertEqual(set(p11[p11.base != "pet_resnet18"].verdict), {"not testable"})
        self.assertTrue((p11[p11.road_metric == "ROAD LeRF"].role == "secondary (no verdict)").all())

    def test_not_testable_when_black_never_over_detects(self):
        pet = ("pet_resnet18", "patch")
        p11 = dose_analysis.road_p11(settings_for({}, keys=[pet]), TITLES)
        row = p11[(p11.base == "pet_resnet18") & (p11.road_metric == "ROAD MoRF")].iloc[0]
        self.assertEqual(row.verdict, "not testable")
        self.assertEqual(row.reason, "FPR(black) is 0")


if __name__ == "__main__":
    unittest.main()
