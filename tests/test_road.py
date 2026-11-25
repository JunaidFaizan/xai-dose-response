import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dose_response
from xai import dose_analysis
from xai.metrics import FAITHFULNESS_METRICS, ROAD_METRICS
from xai.road import (ROAD_GRID, noisy_linear_impute, removed_pixels, road_auc, road_noise_seed)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "road_nli_fixture.npz"
KEYS = ["base", "shortcut", "dose", "seed"]
MAPS = ("shortcut", "object", "object_matched", "placebo", "random")
DOSES, SEEDS, IMAGES, ROAD_N = (-0.5, 0.0, 0.5), (1, 2), 8, 6


class NoisyLinearImputationTest(unittest.TestCase):
    def test_matches_reference_output(self):
        fixture = np.load(FIXTURE)
        self.assertEqual(int(fixture["cases"]), 4)
        for i in range(int(fixture["cases"])):
            keep = fixture[f"mask_{i}"]  # reference convention: 0 = impute
            out = noisy_linear_impute(fixture[f"image_{i}"], keep == 0, noise=0.0)
            self.assertEqual(out.dtype, torch.float32)
            np.testing.assert_allclose(out.numpy(), fixture[f"output_{i}"], atol=1e-5, rtol=0)

    def test_noise_on_removed_pixels_only(self):
        rng = np.random.default_rng(1)
        image = torch.tensor(rng.random((3, 64, 64), dtype=np.float32))
        removed = rng.random((64, 64)) < 0.5
        clean = noisy_linear_impute(image, removed, noise=0.0)
        noisy = noisy_linear_impute(image, removed, noise=0.01, generator=torch.Generator().manual_seed(5))
        again = noisy_linear_impute(image, removed, noise=0.01, generator=torch.Generator().manual_seed(5))
        keep = torch.from_numpy(~removed)
        self.assertTrue(torch.equal(noisy[:, keep], image[:, keep]))
        self.assertTrue(torch.equal(noisy, again))
        diff = (noisy - clean)[:, torch.from_numpy(removed)]
        self.assertTrue((diff != 0).all())
        self.assertAlmostEqual(float(diff.std()), 0.01, delta=5e-4)

    def test_not_clipped_and_nothing_removed(self):
        image = torch.ones(1, 16, 16)
        removed = np.zeros((16, 16), dtype=bool)
        removed[4:12, 4:12] = True
        out = noisy_linear_impute(image, removed, noise=0.01, generator=torch.Generator().manual_seed(0))
        self.assertTrue((out > 1).any())
        copy = noisy_linear_impute(image, np.zeros((16, 16), dtype=bool))
        self.assertTrue(torch.equal(copy, image))
        copy[0, 0, 0] = 5.0
        self.assertEqual(float(image[0, 0, 0]), 1.0)

    def test_single_channel_constant_border(self):
        # A removed pixel surrounded by known pixels of value v is imputed as v.
        image = torch.full((1, 5, 5), 0.3)
        removed = np.zeros((5, 5), dtype=bool)
        removed[0, 0] = removed[2, 2] = True
        out = noisy_linear_impute(image, removed, noise=0.0)
        np.testing.assert_allclose(out.numpy(), image.numpy(), atol=1e-6)

    def test_noise_seed(self):
        self.assertEqual(road_noise_seed("m", 3), road_noise_seed("m", 3))
        self.assertNotEqual(road_noise_seed("m", 3), road_noise_seed("m", 4))
        self.assertNotEqual(road_noise_seed("m", 3), road_noise_seed("n", 3))


class RemovalOrderTest(unittest.TestCase):
    def test_morf_and_lerf_sets(self):
        for n in (10, 784):
            order = np.random.default_rng(n).permutation(n)
            for q in ROAD_GRID:
                k = int(n * q)
                morf, lerf = removed_pixels(order, q, "road_morf"), removed_pixels(order, q, "road_lerf")
                self.assertEqual(len(morf), k)
                self.assertEqual(len(lerf), k)
                np.testing.assert_array_equal(morf, order[:k])
                np.testing.assert_array_equal(lerf, order[n - k:])
                if 2 * k <= n:
                    self.assertFalse(set(morf) & set(lerf))
            self.assertEqual(len(removed_pixels(order, 0.0, "road_morf")), 0)
            self.assertEqual(len(removed_pixels(order, 0.0, "road_lerf")), 0)
        with self.assertRaises(ValueError):
            removed_pixels(np.arange(4), 0.5, "deletion_zero")

    def test_grid_and_area(self):
        self.assertEqual(ROAD_GRID, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9))
        self.assertAlmostEqual(road_auc([0.7] * 8), 0.7, places=12)
        self.assertAlmostEqual(road_auc(list(ROAD_GRID)), 0.45, places=12)  # integral of q over [0, 0.9] / 0.9
        self.assertAlmostEqual(road_auc([1 - q for q in ROAD_GRID]), 0.55, places=12)

    def test_metric_flags(self):
        self.assertEqual(ROAD_METRICS, {"road_morf": ("ROAD (MoRF)", False), "road_lerf": ("ROAD (LeRF)", True)})
        self.assertNotIn("road_morf", FAITHFULNESS_METRICS)
        self.assertFalse(dose_analysis._higher("road_morf"))
        self.assertTrue(dose_analysis._higher("road_lerf"))
        self.assertTrue(dose_analysis._higher("insertion_blur"))


def synthetic(seed=0):
    """Models, oracle (4 existing metrics x 5 maps, 8 images) and ROAD rows
    (2 ROAD metrics x 5 maps, the first 6 images); the shortcut map wins more
    often as the dose grows."""
    rng = np.random.default_rng(seed)
    index = rng.permutation(np.arange(100, 100 + IMAGES))
    models, oracle, road = [], [], []
    for dose in DOSES:
        for s in SEEDS:
            key = {"base": "pet_resnet18", "shortcut": "patch", "dose": dose, "seed": s}
            reliance = 50 + 80 * dose + s
            models.append({**key, "reliance": reliance, "reliance_corrected": reliance, "model_set": "original"})
            for n, i in enumerate(index):
                follows = rng.random() < (dose + 0.5)
                row = {**key, "index": int(i), "label": int(i % 2)}
                for metric, (_, _, _, higher) in FAITHFULNESS_METRICS.items():
                    for which in MAPS:
                        row[f"{metric}_{which}"] = 0.5 + 0.05 * rng.normal()
                    row[f"{metric}_shortcut"] = 0.2 if follows != higher else 0.8
                oracle.append(row)
                if n < ROAD_N:
                    road_row = {**key, "index": int(i), "label": int(i % 2)}
                    for metric, (_, higher) in ROAD_METRICS.items():
                        for which in MAPS:
                            road_row[f"{metric}_{which}"] = 0.5 + 0.05 * rng.normal()
                        road_row[f"{metric}_shortcut"] = 0.1 if follows != higher else 0.9
                    road.append(road_row)
    return pd.DataFrame(models), pd.DataFrame(oracle), pd.DataFrame(road)


class RoadAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="test_road_"))
        (self.tmp / "per_model").mkdir()
        self.saved_out = dose_response.OUT
        dose_response.OUT = self.tmp
        self.models, self.oracle, self.road = synthetic()

    def tearDown(self):
        dose_response.OUT = self.saved_out
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_road(self, frame):
        for key, group in frame.groupby(KEYS):
            group.to_csv(self.tmp / "per_model" / f"{dose_response.model_name(*key)}_road.csv", index=False)

    def test_no_road_scores_writes_nothing(self):
        out = self.tmp / "out"
        out.mkdir()
        self.assertEqual(dose_response.road_analysis(self.oracle, self.models, 20, out), [])
        self.assertEqual(list(out.iterdir()), [])

    def test_tables_on_the_same_images(self):
        self.write_road(self.road)
        out = self.tmp / "out"
        out.mkdir()
        lines = dose_response.road_analysis(self.oracle, self.models, 20, out)
        self.assertEqual(lines[0], "## Exact ROAD (SD3, amendment A3)")
        files = {"road_models.csv", "preference_road.csv", "responsiveness_corrected_road.csv",
                 "overdetection_road.csv", "road_p10.csv", "road_p11.csv"}
        self.assertEqual({p.name for p in out.iterdir()}, files)
        road_models = pd.read_csv(out / "road_models.csv")
        self.assertEqual(len(road_models), len(self.models))
        self.assertTrue((road_models.images == ROAD_N).all())
        pref = pd.read_csv(out / "preference_road.csv")
        titles = {**dose_response.METRIC_TITLES, **dose_response.ROAD_TITLES}
        self.assertEqual(set(pref.metric), set(titles.values()))
        # Existing metrics: the preference on the first ROAD_N images only.
        subset = self.oracle.merge(self.road[KEYS + ["index"]], on=KEYS + ["index"])
        expected = dose_response.preference_table(subset, "shortcut", "object")
        got = pref[pref.comparison == "shortcut vs full object"]
        for metric in FAITHFULNESS_METRICS:
            rows = got[got.metric == titles[metric]].merge(expected[KEYS + [metric]], on=KEYS)
            self.assertEqual(len(rows), len(self.models))
            np.testing.assert_allclose(rows.pref, rows[metric].round(4), atol=1e-12)
        resp = pd.read_csv(out / "responsiveness_corrected_road.csv")
        self.assertEqual(set(resp.metric), set(titles.values()))
        over = pd.read_csv(out / "overdetection_road.csv")
        self.assertEqual(len(over), len(titles))

    def test_partial_road_scores(self):
        first = self.models.iloc[0]
        self.write_road(self.road[(self.road.dose == first.dose) & (self.road.seed == first.seed)])
        out = self.tmp / "out"
        out.mkdir()
        lines = dose_response.road_analysis(self.oracle, self.models, 20, out)
        self.assertTrue(lines)
        self.assertEqual(len(pd.read_csv(out / "road_models.csv")), 1)

    def test_unmatched_road_row_raises(self):
        road = self.road.copy()
        road.loc[0, "index"] = 999
        self.write_road(road)
        out = self.tmp / "out"
        out.mkdir()
        with self.assertRaises(ValueError):
            dose_response.road_analysis(self.oracle, self.models, 20, out)

    def test_default_metrics_unchanged(self):
        comparisons = dose_response.COMPARISONS
        settings = dose_analysis.build_settings(self.oracle, None, self.models, 20, 0, 1e-4)
        titles = dose_response.METRIC_TITLES
        pd.testing.assert_frame_equal(
            dose_analysis.responsiveness_corrected(settings, comparisons, titles),
            dose_analysis.responsiveness_corrected(settings, comparisons, titles, metrics=FAITHFULNESS_METRICS))
        pd.testing.assert_frame_equal(
            dose_analysis.overdetection(settings, comparisons, titles),
            dose_analysis.overdetection(settings, comparisons, titles, metrics=FAITHFULNESS_METRICS))


if __name__ == "__main__":
    unittest.main()
