"""SD4 watermark shortcut (amendment A3): marks, rendering, masks and the base check."""

import unittest

import numpy as np
import torch

from xai.data import (WATERMARK_MASK_FILE, DecoyDataset, border_positions, shortcut_split,
                      watermark_masks)


class SyntheticBase:
    """3x128x128 images in [0, 1] with alternating labels (no Pet data needed)."""

    def __init__(self, n=8):
        g = torch.Generator().manual_seed(0)
        self.images = torch.rand((n, 3, 128, 128), generator=g)
        self.targets = np.arange(n) % 2

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.images[i], int(self.targets[i])


def dataset(mode="aligned"):
    base = SyntheticBase()
    masks = np.ones((len(base), 128, 128), dtype=bool)  # the whole image is "pet"
    return base, DecoyDataset(base, border_positions(2, 128, 12), mode=mode, shortcut="watermark",
                              patch_size=12, object_masks=masks)


def box_set(rows, cols):
    return {(r, c) for r in rows for c in cols}


class WatermarkTest(unittest.TestCase):
    def test_saved_masks_equal_the_drawn_ones(self):
        saved = np.load(WATERMARK_MASK_FILE)
        self.assertEqual(saved.dtype, np.bool_)
        np.testing.assert_array_equal(saved, watermark_masks())

    def test_mask_geometry(self):
        masks = watermark_masks()
        self.assertEqual(masks.shape, (2, 128, 128))
        self.assertEqual([int(m.sum()) for m in masks], [96, 96])
        self.assertFalse((masks[0] & masks[1]).any())
        box = np.zeros((128, 128), dtype=bool)
        box[116:128, 104:128] = True
        self.assertFalse((masks.any(axis=0) & ~box).any())
        cat = box_set(range(116, 128), list(range(104, 108)) + list(range(124, 128)))
        dog = box_set(list(range(116, 120)) + list(range(124, 128)), range(110, 122))
        self.assertEqual(set(zip(*np.nonzero(masks[0]))), cat)
        self.assertEqual(set(zip(*np.nonzero(masks[1]))), dog)

    def test_rendering(self):
        base, ds = dataset()
        masks = torch.from_numpy(watermark_masks())
        for i in range(len(ds)):
            image, label = ds[i]
            original = base.images[i]
            mask = masks[label].expand(3, -1, -1)
            torch.testing.assert_close(image[mask], 0.6 * original[mask] + 0.4, rtol=0, atol=1e-6)
            self.assertTrue(torch.equal(image[~mask], original[~mask]))
            self.assertTrue(torch.equal(ds.with_shortcut(original, label), image))

    def test_conflicting_shows_the_other_mark(self):
        base, ds = dataset("conflicting")
        masks = watermark_masks()
        for i in range(len(ds)):
            image, label = ds[i]
            other = 1 - label
            self.assertEqual(int(ds.patch_class[i]), other)
            np.testing.assert_array_equal(ds.mask(i), masks[other])
            torch.testing.assert_close(image, ds.with_shortcut(base.images[i], other), rtol=0, atol=0)

    def test_maps(self):
        _, ds = dataset()
        masks = watermark_masks()
        for i in range(len(ds)):
            label = int(ds.labels[i])
            np.testing.assert_array_equal(ds.mask(i), masks[label])
            np.testing.assert_array_equal(ds.mask_for_class(1 - label), masks[1 - label])
            values = ds.object_map(i)
            self.assertTrue((values[masks[label]] == 0).all())
            self.assertTrue((values[~masks[label]] == 1).all())

    def test_only_the_pretrained_pet_base(self):
        for base in ("pet_resnet18", "mnist", "cifar10_resnet18"):
            with self.subTest(base=base), self.assertRaises(ValueError):
                shortcut_split(base, "watermark", "test")

    def test_needs_128px_two_classes(self):
        class Small(SyntheticBase):
            def __getitem__(self, i):
                return self.images[i][:, :32, :32], int(self.targets[i])

        with self.assertRaises(ValueError):
            DecoyDataset(Small(), border_positions(2, 32), shortcut="watermark")


if __name__ == "__main__":
    unittest.main()
