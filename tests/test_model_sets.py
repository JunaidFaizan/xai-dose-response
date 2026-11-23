import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import dose_response
from dose_response import in_model_set, model_set_of

ROOT = Path(__file__).resolve().parents[1]


def run(base, dose, seed, shortcut="patch"):
    return {"base": base, "shortcut": shortcut, "dose": dose, "seed": seed}


class ModelSetTest(unittest.TestCase):
    def test_a2_final_doses(self):
        for dose in (-0.05, -0.03, -0.02, -0.01):
            self.assertEqual(model_set_of(run("cifar10_resnet18", dose, 42)), "A2")
            self.assertEqual(model_set_of(run("cifar10", dose, 1)), "A2")
        for dose in (-0.5, -0.2, 0.0, 0.1, 0.25, 0.4):
            self.assertEqual(model_set_of(run("pet_resnet18_pretrained", dose, 2)), "A2")
        self.assertEqual(model_set_of(run("pet_resnet18", -0.15, 1)), "A2")

    def test_sd1_pilots(self):
        for dose in (0.0, -0.5):
            self.assertEqual(model_set_of(run("pet_resnet18_pretrained", dose, 7)), "A2_pilot")
            self.assertFalse(in_model_set(run("pet_resnet18_pretrained", dose, 7), "all"))

    def test_unplanned_models_are_unregistered(self):
        cases = [run("pet_resnet18", -0.05, 42),            # needs a new amendment
                 run("pet_resnet18", -0.15, 7),             # A2 dose, unplanned seed
                 run("cifar10_resnet18", -0.075, 42),       # another A2 rule's dose set
                 run("cifar10", -0.03, 2),                  # small CNN uses seeds 42 and 1
                 run("pet_resnet18_pretrained", -0.4, 42),  # the other SD1 dose set
                 run("pet_resnet18_pretrained", 0.1, 7),    # pilot seed, not a pilot dose
                 run("mnist", 0.3, 42),
                 run("cifar10", 0.0, 42, shortcut="texture"),
                 run("new_base", 0.0, 42)]
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(model_set_of(case), "unregistered")
                for model_set in ("all", "original", "pre_a2", "A1", "A2"):
                    self.assertFalse(in_model_set(case, model_set))

    def test_pre_a2_models_keep_their_sets(self):
        """The 234 committed pre-A2 models: 192 original and 42 A1, as recorded in models.csv."""
        models = pd.read_csv(ROOT / "results" / "dose_response" / "models.csv")
        models = models[models.model_set.isin(["original", "A1"])]
        got = [model_set_of(row) for _, row in models.iterrows()]
        self.assertEqual(got, list(models.model_set))
        self.assertEqual(got.count("original"), 192)
        self.assertEqual(got.count("A1"), 42)

    def test_sd4_pilots(self):
        for dose in (0.0, 0.25):
            case = run("pet_resnet18_pretrained", dose, 7, shortcut="watermark")
            self.assertEqual(model_set_of(case), "A3_pilot")
            self.assertFalse(in_model_set(case, "all"))
            self.assertTrue(in_model_set(case, "A3_pilot"))

    def test_sd4_confirmatory_unregistered_until_the_completion_note(self):
        self.assertIsNone(dose_response.A3_SD4_RULE)
        self.assertEqual(dose_response.a3_design(), {})
        for doses in dose_response.A3_SD4_DOSE_SETS.values():
            for dose in doses:
                for seed in (42, 1, 2):
                    case = run("pet_resnet18_pretrained", dose, seed, shortcut="watermark")
                    self.assertEqual(model_set_of(case), "unregistered")
                    self.assertFalse(in_model_set(case, "all"))

    def test_sd4_confirmatory_after_the_completion_note(self):
        every = sorted({d for doses in dose_response.A3_SD4_DOSE_SETS.values() for d in doses})
        for key, doses in dose_response.A3_SD4_DOSE_SETS.items():
            with self.subTest(rule=key), mock.patch.object(dose_response, "A3_SD4_RULE", key):
                for dose in every:
                    for seed in (42, 1, 2):
                        case = run("pet_resnet18_pretrained", dose, seed, shortcut="watermark")
                        expected = "A3" if np.isclose(dose, doses).any() else "unregistered"
                        self.assertEqual(model_set_of(case), expected)
                        self.assertEqual(in_model_set(case, "all"), expected == "A3")
                    if not np.isclose(dose, [0.0, 0.25]).any():  # seed 7 outside the pilot doses
                        case = run("pet_resnet18_pretrained", dose, 7, shortcut="watermark")
                        self.assertEqual(model_set_of(case), "unregistered")
                self.assertEqual(sum(model_set_of(run("pet_resnet18_pretrained", d, s, shortcut="watermark"))
                                     == "A3" for d in every for s in (42, 1, 2)), 18)
                # patch models on the pretrained network keep their A2 sets
                for dose in (-0.5, -0.2, 0.0, 0.1, 0.25, 0.4):
                    self.assertEqual(model_set_of(run("pet_resnet18_pretrained", dose, 42)), "A2")
                for dose in (0.0, -0.5):
                    self.assertEqual(model_set_of(run("pet_resnet18_pretrained", dose, 7)), "A2_pilot")
                self.assertEqual(model_set_of(run("pet_resnet18_pretrained", -0.4, 42)), "unregistered")
                self.assertEqual(model_set_of(run("pet_resnet18", 0.0, 42, shortcut="watermark")),
                                 "unregistered")

    def test_sd4_dose_and_stop_rules(self):
        self.assertEqual(dose_response.a3_dose_set(60.0), [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
        self.assertEqual(dose_response.a3_dose_set(59.99), [-0.5, -0.2, 0.0, 0.1, 0.25, 0.4])
        self.assertTrue(dose_response.a3_sd4_stopped(19.99))
        self.assertFalse(dose_response.a3_sd4_stopped(20.0))

    def test_cli_refuses_watermark_on_other_bases(self):
        argv = ["dose_response.py", "train", "--bases", "mnist", "--shortcuts", "watermark"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(dose_response, "train") as train, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            dose_response.main()
        train.assert_not_called()

    def test_final_doses_follow_the_rules(self):
        self.assertEqual(dose_response.a2_dose_set("cifar10_resnet18", 48.18), [-0.05, -0.03, -0.02, -0.01])
        self.assertEqual(dose_response.a2_dose_set("pet_resnet18_pretrained", 31.96),
                         [-0.5, -0.2, 0.0, 0.1, 0.25, 0.4])


if __name__ == "__main__":
    unittest.main()
