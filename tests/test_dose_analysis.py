import unittest

import numpy as np
import pandas as pd
import torch

from xai import dose_analysis, stats
from xai.data import DecoyDataset
from xai.explain import METHODS
from xai.metrics import FAITHFULNESS_METRICS

COMPARISONS = [
    ("shortcut", "object", "shortcut vs full object", "", ""),
    ("random", "object", "random vs full object", "", ""),
    ("shortcut", "random", "shortcut vs random", "", ""),
]
TITLES = {name: name for name in FAITHFULNESS_METRICS}
METHOD_TITLES = {name: name for name in METHODS}
DOSES, SEEDS, IMAGES = (-0.5, 0.0, 0.5), (1, 2), 8


def synthetic(base="pet_resnet18", seed=0):
    '''Models, per-image oracle scores (deletion: lower AUC wins), method scores and
    behaviour rows; image indices are shuffled within each model.'''
    rng = np.random.default_rng(seed)
    models, oracle, methods, behaviour = [], [], [], []
    for dose in DOSES:
        for s in SEEDS:
            key = {"base": base, "shortcut": "patch", "dose": dose, "seed": s}
            reliance = 50 + 80 * dose + s
            models.append({**key, "reliance": reliance, "reliance_corrected": reliance,
                           "test_accuracy_no_shortcut": 90.0 - 10 * dose})
            index = rng.permutation(np.arange(100, 100 + IMAGES))
            follows = rng.integers(0, 2, IMAGES)
            for i, f in zip(index, follows):
                row = {**key, "index": int(i)}
                for metric, (_, _, _, higher) in FAITHFULNESS_METRICS.items():
                    shortcut = 0.2 if (f == 1) != higher else 0.8  # the metric rates shortcut better iff f
                    row[f"{metric}_shortcut"], row[f"{metric}_object"] = shortcut, 0.5
                    row[f"{metric}_random"] = 0.5 + 0.01 * rng.normal()
                oracle.append(row)
                for method in METHODS:
                    methods.append({**key, "index": int(i), "method": method, "shortcut_mass": reliance + i})
                conflict = 1
                behaviour.append({**key, "index": int(i), "label": int(i % 2), "conflict_class": conflict,
                                  "pred_conflicting": conflict if f else 0,
                                  "pred_none": conflict if i == 100 else 0,
                                  "follows_shortcut": int(f), "shortcut_dominant": int(f)})
    return pd.DataFrame(models), pd.DataFrame(oracle), pd.DataFrame(methods), pd.DataFrame(behaviour)


def class_tied_synthetic():
    '''Two classes with the label tied to the class (PneumoniaMNIST-patch-like): class 0
    images are mostly followed, class 1 mostly not, and class 1 has the larger AUC gap.
    Within each class the gap ranks followed images first (stratified AUROC 1), but over
    both classes the followed images have the smaller gaps (unstratified AUROC 7/16).'''
    models, oracle, behaviour = [], [], []
    follows = {100: 1, 102: 1, 104: 1, 106: 0, 101: 0, 103: 0, 105: 0, 107: 1}
    for dose in DOSES:
        for s in SEEDS:
            key = {"base": "pneumonia_mnist", "shortcut": "patch", "dose": dose, "seed": s}
            reliance = 50 + 80 * dose + s
            models.append({**key, "reliance": reliance, "reliance_corrected": reliance,
                           "test_accuracy_no_shortcut": 90.0})
            for i, f in follows.items():
                cls = i % 2
                gap = (0.1 if f else 0.0) + 0.4 * cls
                row = {**key, "index": i}
                for metric, (_, _, _, higher) in FAITHFULNESS_METRICS.items():
                    row[f"{metric}_shortcut"] = 0.5 + gap if higher else 0.5 - gap
                    row[f"{metric}_object"] = 0.5
                    row[f"{metric}_random"] = 0.5
                oracle.append(row)
                conflict = 1 - cls
                behaviour.append({**key, "index": i, "label": cls, "conflict_class": conflict,
                                  "pred_conflicting": conflict if f else cls, "pred_none": cls,
                                  "follows_shortcut": f, "shortcut_dominant": f})
    return pd.DataFrame(models), pd.DataFrame(oracle), pd.DataFrame(behaviour)


def setting(reps=20):
    models, oracle, methods, behaviour = synthetic()
    settings = dose_analysis.build_settings(oracle, methods, models, reps, 0, 1e-4)
    return settings, behaviour


class SettingTests(unittest.TestCase):
    '''Setting indexing and resampling'''

    def test_matrix_alignment(self):
        '''rows sorted by (dose, seed), images by index, whatever the input order'''
        (s,), _ = setting()
        self.assertEqual((s.n_models, s.n_images), (6, IMAGES))
        np.testing.assert_array_equal(s.indices[0], np.arange(100, 100 + IMAGES))
        self.assertTrue((s.indices == s.indices[0]).all())
        np.testing.assert_array_equal(s.dose, np.repeat(DOSES, len(SEEDS)))

    def test_resample_indexing(self):
        '''resample(values, mean)[r, m] equals values[P[r, m], I[r, m]].mean()'''
        (s,), _ = setting()
        values = np.random.default_rng(1).normal(size=(s.n_models, s.n_images))
        out = s.resample(values, np.mean)
        for r in (0, 7, 19):
            for m in range(s.n_models):
                self.assertAlmostEqual(out[r, m], values[s.P[r, m], s.I[r, m]].mean())

    def test_unequal_images_raise(self):
        models, oracle, methods, _ = synthetic()
        with self.assertRaises(ValueError):
            dose_analysis.build_settings(oracle.iloc[1:], methods, models, 5, 0, 1e-4)


class BehaviourTests(unittest.TestCase):
    '''MD4 label derivation and label/score alignment'''

    def test_causal_label(self):
        frame = pd.DataFrame({"conflict_class": [1, 1, 1, 1], "pred_conflicting": [1, 1, 0, 0],
                              "pred_none": [0, 1, 0, 1]})
        out = dose_analysis.add_causal_label(frame)
        '''pred_none on the shortcut's class makes the image undetermined, whatever the moved pass says'''
        np.testing.assert_array_equal(out.follows_shortcut_causal.to_numpy()[[0, 2]], [1, 0])
        self.assertTrue(np.isnan(out.follows_shortcut_causal.to_numpy()[[1, 3]]).all())

    def test_alignment_gives_perfect_auroc(self):
        '''the metric rates the shortcut better exactly on followed images: AUROC 1
        (pooled and within model), even with shuffled behaviour rows'''
        settings, behaviour = setting()
        behaviour = behaviour.sample(frac=1, random_state=0)
        _, summary, bands = dose_analysis.behaviour_tables(settings, behaviour, COMPARISONS, TITLES)
        primary = summary[summary.label == dose_analysis.PRIMARY_LABEL]
        self.assertEqual(len(primary), len(FAITHFULNESS_METRICS))
        np.testing.assert_allclose(primary.auroc_pooled, 1.0)
        np.testing.assert_allclose(primary.median_model_auroc, 1.0)
        np.testing.assert_allclose(primary.auroc_pooled_strat, 1.0)
        np.testing.assert_allclose(primary.median_model_auroc_strat, 1.0)
        # index 100 is predicted as the shortcut's class without shortcut: undetermined everywhere
        np.testing.assert_allclose(primary.undetermined_rate, 1 / IMAGES)
        used = bands[bands.label == dose_analysis.PRIMARY_LABEL]
        self.assertEqual(int((used.n_images + used.n_undetermined).sum()),
                         len(FAITHFULNESS_METRICS) * 6 * IMAGES)

    def test_class_tied_label_is_stratified(self):
        '''label tied to the class: unstratified AUROC 7/16 (below 0.5, two-sided, direction
        "below 0.5"); class-stratified AUROC 1 with a one-sided p; the bootstrap keeps classes'''
        models, oracle, behaviour = class_tied_synthetic()
        settings = dose_analysis.build_settings(oracle, None, models, 50, 0, 1e-4)
        model_frame, summary, bands = dose_analysis.behaviour_tables(settings, behaviour, COMPARISONS, TITLES)
        primary = summary[summary.label == dose_analysis.PRIMARY_LABEL]
        np.testing.assert_allclose(primary.median_model_auroc, 7 / 16)
        np.testing.assert_allclose(primary.auroc_pooled, 7 / 16)
        self.assertTrue((primary.direction_within == "below 0.5").all())
        np.testing.assert_allclose(primary.median_model_auroc_strat, 1.0)
        np.testing.assert_allclose(primary.auroc_pooled_strat, 1.0)
        self.assertTrue((primary.direction_within_strat == "above 0.5").all())
        self.assertTrue((primary.models_with_both_labels_in_a_class == 6).all())
        # every resampled class still ranks followed images first, so all replicates stay >= 0.5
        self.assertTrue((primary.median_model_auroc_strat_ci_low >= 0.5).all())
        self.assertTrue((primary.p_auroc_within_strat < 0.5).all())
        np.testing.assert_allclose(bands[bands.label == dose_analysis.PRIMARY_LABEL].auroc_strat, 1.0)
        column = f"auroc_strat_{dose_analysis.PRIMARY_LABEL}_{next(iter(FAITHFULNESS_METRICS))}_full"
        np.testing.assert_allclose(model_frame[column], 1.0)
        resp = pd.DataFrame({"headline": [True], "p_rho_rc": [0.01]})
        _, headline = dose_analysis.headline_holm(resp, summary)
        self.assertEqual(int(headline.p_holm_headline_within_strat.notna().sum()), len(FAITHFULNESS_METRICS))

    def test_mismatched_labels_raise(self):
        settings, behaviour = setting()
        behaviour = behaviour.copy()
        behaviour.loc[behaviour.index[0], "index"] = 999
        with self.assertRaises(ValueError):
            dose_analysis.behaviour_tables(settings, behaviour, COMPARISONS, TITLES)

    def test_masked_auroc_rows(self):
        '''entries outside valid do not change the AUROC'''
        scores = np.array([[0.1, 0.9, 0.2, 0.8, 5.0]])
        labels = np.array([[0, 1, 0, 1, 0]])
        valid = np.array([[True, True, True, True, False]])
        self.assertEqual(stats.auroc_rows(scores, labels, valid)[0], 1.0)
        self.assertLess(stats.auroc_rows(scores, labels)[0], 1.0)


class SwitchTests(unittest.TestCase):
    def test_bracketing(self):
        '''MNIST-patch-like gap 22..45: a switch at 32 is identified from one edge only'''
        x = [0, 5, 10, 22, 45, 80]
        self.assertEqual(stats.switch_identification(32.0, x)[1], "identified")
        below, above, bracketed = stats.switch_bracketing(32.0, x)
        self.assertEqual((below, above, bracketed), (10.0, 13.0, False))
        self.assertTrue(stats.switch_bracketing(30.0, [22, 35])[2])

    def test_edge_status(self):
        switch, status, *_ , edge = dose_analysis.switch_status([0, 10, 20], [0.4, 0.6, 0.7])
        self.assertAlmostEqual(switch, 5.0)
        self.assertTrue(edge)
        self.assertEqual(status, dose_analysis.EDGE_STATUS)

    def test_z_p(self):
        boot = np.random.default_rng(0).normal(0.8, 0.05, 2000)
        self.assertLess(stats.bootstrap_z_p(0.8, boot, 0.5), 1e-6)
        self.assertGreater(stats.bootstrap_z_p(0.5, boot, 0.5), 0.99)
        '''one-sided (H1: > null): an estimate far below the null is not significant'''
        low = np.random.default_rng(0).normal(0.2, 0.05, 2000)
        self.assertLess(stats.bootstrap_z_p(0.2, low, 0.5), 1e-6)
        self.assertGreater(stats.bootstrap_z_p(0.2, low, 0.5, "greater"), 0.999)
        self.assertLess(stats.bootstrap_z_p(0.8, boot, 0.5, "greater"), 1e-6)


class PetTests(unittest.TestCase):
    def test_pet_preregistered(self):
        '''one row per metric for P1, P2 (+ contrast), P3 (low + rho) and two P4 methods; verdicts kept'''
        settings, _ = setting()
        frame = dose_analysis.pet_preregistered(settings, COMPARISONS, TITLES, METHOD_TITLES)
        counts = frame.prediction.value_counts().to_dict()
        n = len(FAITHFULNESS_METRICS)
        self.assertEqual(counts, {"P1": n, "P2": n + 1, "P3": 2 * n, "P4": 2})
        for p, verdict in dose_analysis.PET_VERDICTS.items():
            self.assertTrue((frame[frame.prediction == p].original_verdict == verdict).all())
        p4 = frame[frame.prediction == "P4"]
        np.testing.assert_allclose(p4.estimate, 1.0)  # mass rises with reliance


class BehaviourOneTests(unittest.TestCase):
    '''behaviour_one with a stub model on a tiny synthetic patch dataset'''

    def test_rows(self):
        import dose_response

        class Base(torch.utils.data.Dataset):
            labels = np.array([0, 1, 0, 1, 0, 1])

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, i):
                return torch.zeros(1, 8, 8), int(self.labels[i])

        positions = [(0, 0), (4, 4)]
        sets = tuple(DecoyDataset(Base(), positions, mode=m, patch_size=2) for m in ("aligned", "conflicting", "none"))

        class PatchReader(torch.nn.Module):
            '''predicts class 1 iff the class-1 patch at (4, 4) is present; else class 0'''

            def forward(self, x):
                on = (x[:, 0, 4, 4] > 0.5).float()
                logits = torch.stack([1 - on, on], dim=1) * 5
                return torch.log_softmax(logits, dim=1)

        original = dose_response.load
        dose_response.load = lambda *a: PatchReader()
        try:
            rows = pd.DataFrame(dose_response.behaviour_one("toy", "patch", 0.0, 1, 6, sets))
        finally:
            dose_response.load = original
        self.assertEqual(len(rows), 6)
        for _, r in rows.iterrows():
            self.assertNotEqual(sets[0].labels[r.swap_index], r.label)  # partner of another class
            self.assertEqual(r.conflict_class, 1 - r.label)
            self.assertEqual(r.follows_shortcut, 1)  # the stub reads only the shortcut
            self.assertEqual(r.pred_none, 0)
        x = sets[0].with_shortcut(torch.zeros(1, 8, 8), 1)
        self.assertEqual(float(x[0, 4:6, 4:6].min()), 1.0)
        self.assertEqual(float(x.sum()), 4.0)
        causal = dose_analysis.add_causal_label(rows).follows_shortcut_causal
        # label-1 images: conflict class 0 = pred_none, so undetermined; label-0 images: causal 1
        np.testing.assert_array_equal(np.isnan(causal), rows.label == 1)


if __name__ == "__main__":
    unittest.main()
