import unittest
from unittest.mock import patch

import numpy as np

from spatial_denoise.core import (
    RunConfig, add_noise, edge_f1, mean_filter, median_filter, mse, psnr, run_pipeline, sobel,
)


def oracle(image, size, reducer):
    padded = np.pad(image, size // 2, mode="edge")
    result = np.empty_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result[y, x] = np.rint(reducer(padded[y:y + size, x:x + size]))
    return result


class FilterTests(unittest.TestCase):
    def test_rgb_filters_against_oracle_and_sobel_luminance(self):
        image = np.random.default_rng(8).integers(0, 256, (7, 9, 3), dtype=np.uint8)
        for function, reducer in ((mean_filter, np.mean), (median_filter, np.median)):
            expected = np.stack([oracle(image[..., c], 3, reducer) for c in range(3)], axis=-1)
            np.testing.assert_array_equal(function(image), expected)
        red = np.zeros((5, 5, 3), np.uint8)
        red[:, 2:, 0] = 255
        gray = np.zeros((5, 5), np.uint8)
        gray[:, 2:] = 255
        np.testing.assert_allclose(sobel(red), sobel(gray) * 0.299, atol=1e-12)

    def test_rgb_pipeline_preserves_original_and_salt_pepper_pixels(self):
        image = np.full((20, 20, 3), [40, 100, 200], np.uint8)
        result = run_pipeline(image, RunConfig(noise="salt_pepper", amount=0.5))
        np.testing.assert_array_equal(result.images["original"], image)
        noisy = result.images["noisy"]
        changed = np.any(noisy != image, axis=-1)
        self.assertTrue(changed.any())
        self.assertTrue(np.all(np.all(noisy[changed] == 0, axis=-1) | np.all(noisy[changed] == 255, axis=-1)))
        for output in result.images.values():
            self.assertEqual(output.shape, image.shape)
        self.assertEqual(result.gradients["noisy"].shape, image.shape[:2])
        self.assertAlmostEqual(result.metrics[0]["mse"], np.mean((noisy.astype(float) - image) ** 2))

    def test_against_independent_loop_oracle(self):
        rng = np.random.default_rng(18)
        for shape in ((1, 1), (1, 9), (8, 1), (9, 11), (75, 14)):
            image = rng.integers(0, 256, shape, dtype=np.uint8)
            for size in (3, 5, 7):
                with self.subTest(shape=shape, size=size):
                    np.testing.assert_array_equal(mean_filter(image, size), oracle(image, size, np.mean))
                    np.testing.assert_array_equal(median_filter(image, size), oracle(image, size, np.median))

    def test_constant_and_no_input_mutation(self):
        image = np.full((8, 11), 137, dtype=np.uint8)
        for function in (mean_filter, median_filter):
            np.testing.assert_array_equal(function(image, 7), image)
        np.testing.assert_array_equal(image, np.full((8, 11), 137, dtype=np.uint8))

    def test_impulse(self):
        image = np.full((5, 5), 10, dtype=np.uint8)
        image[2, 2] = 255
        self.assertEqual(median_filter(image)[2, 2], 10)
        self.assertEqual(mean_filter(image)[2, 2], 37)

    def test_padding_values_at_corner(self):
        image = np.array([[0, 90], [180, 255]], dtype=np.uint8)
        self.assertEqual(mean_filter(image)[0, 0], round((0 * 4 + 90 * 2 + 180 * 2 + 255) / 9))
        self.assertEqual(median_filter(image)[0, 0], 90)

    def test_validation(self):
        image = np.zeros((4, 4), dtype=np.uint8)
        for bad in (0, 1, 2, 32, 3.0, True):
            for function in (mean_filter, median_filter):
                with self.assertRaises(ValueError):
                    function(image, bad)
        for bad in (np.zeros((0, 2), np.uint8), np.zeros((3, 3)), np.zeros((3, 3, 4), np.uint8)):
            with self.assertRaises(ValueError):
                mean_filter(bad)


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((160, 160), 128, dtype=np.uint8)

    def test_zero_noise_and_no_mutation(self):
        for config in (RunConfig(sigma=0), RunConfig(noise="salt_pepper", amount=0)):
            output = add_noise(self.image, config)
            np.testing.assert_array_equal(output, self.image)
            output[0, 0] = 0
            self.assertEqual(self.image[0, 0], 128)

    def test_reproducible_seed_and_different_seeds(self):
        for noise in ("gaussian", "salt_pepper"):
            a = add_noise(self.image, RunConfig(noise=noise, seed=42))
            b = add_noise(self.image, RunConfig(noise=noise, seed=42))
            c = add_noise(self.image, RunConfig(noise=noise, seed=43))
            np.testing.assert_array_equal(a, b)
            self.assertFalse(np.array_equal(a, c))

    def test_salt_pepper_distribution(self):
        output = add_noise(self.image, RunConfig(noise="salt_pepper", amount=0.1))
        self.assertLess(abs(np.mean(output == 0) - 0.05), 0.01)
        self.assertLess(abs(np.mean(output == 255) - 0.05), 0.01)
        self.assertTrue(set(np.unique(output)) <= {0, 128, 255})
        full = add_noise(self.image, RunConfig(noise="salt_pepper", amount=1))
        self.assertTrue(set(np.unique(full)) <= {0, 255})

    def test_gaussian_distribution_and_clipping(self):
        output = add_noise(self.image, RunConfig(sigma=20)).astype(float) - 128
        self.assertLess(abs(output.mean()), 0.5)
        self.assertLess(abs(output.std() - 20), 0.6)
        output = add_noise(np.zeros((50, 50), np.uint8), RunConfig(sigma=100))
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output.min(), 0)

    def test_invalid_config(self):
        for config in (RunConfig(noise="bad"), RunConfig(sigma=float("nan")), RunConfig(amount=1.1),
                       RunConfig(seed=-1), RunConfig(seed=2**32), RunConfig(seed=1.5), RunConfig(edge_threshold=0)):
            with self.assertRaises(ValueError):
                add_noise(self.image, config)


class MetricAndSobelTests(unittest.TestCase):
    def test_metrics_known_values_uint8(self):
        zeros, full = np.zeros((4, 4), np.uint8), np.full((4, 4), 255, np.uint8)
        self.assertEqual(mse(zeros, full), 65025)
        self.assertEqual(psnr(zeros, full), 0)
        self.assertEqual(mse(full, full), 0)
        self.assertTrue(np.isposinf(psnr(full, full)))
        self.assertAlmostEqual(psnr(zeros, np.full((4, 4), 10, np.uint8)), 28.130803608679106)
        self.assertEqual(psnr(np.zeros(4), np.ones(4), data_range=1), 0)
        with self.assertRaises(ValueError):
            mse(zeros, np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            mse(np.array([np.nan]), np.array([0]))

    def test_sobel_constant_ramp_and_step(self):
        np.testing.assert_array_equal(sobel(np.full((6, 7), 45, np.uint8)), np.zeros((6, 7)))
        ramp = np.tile(np.arange(7, dtype=np.uint8) * 10, (6, 1))
        np.testing.assert_allclose(sobel(ramp)[1:-1, 1:-1], 10)
        step = np.zeros((8, 8), np.uint8)
        step[:, 4:] = 255
        gradient = sobel(step)
        np.testing.assert_allclose(gradient[:, 3:5], 127.5)
        np.testing.assert_array_equal(gradient[:, :3], 0)
        np.testing.assert_array_equal(gradient[:, 5:], 0)

    def test_edge_f1_conventions(self):
        empty, full = np.zeros((2, 2)), np.full((2, 2), 30.)
        self.assertEqual(edge_f1(empty, empty, 20), 1)
        self.assertEqual(edge_f1(empty, full, 20), 0)
        self.assertEqual(edge_f1(full, full, 20), 1)
        half = full.copy()
        half[0] = 0
        self.assertAlmostEqual(edge_f1(full, half, 20), 2 / 3)

    def test_pipeline_same_noisy_image_and_salt_pepper_improvement(self):
        image = np.full((80, 80), 128, np.uint8)
        config = RunConfig(noise="salt_pepper", amount=0.05)
        result = run_pipeline(image, config)
        np.testing.assert_array_equal(result.images["mean"], mean_filter(result.images["noisy"]))
        np.testing.assert_array_equal(result.images["median"], median_filter(result.images["noisy"]))
        rows = {row["method"]: row for row in result.metrics}
        self.assertLess(rows["median"]["mse"], rows["mean"]["mse"])
        self.assertGreater(rows["median"]["psnr_db"], rows["noisy"]["psnr_db"])


class ExistingNoiseTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.full((15, 17), 128, np.uint8)
        self.noisy = self.clean.copy()
        self.noisy[7, 8] = 255
        self.config = RunConfig(input_mode="existing", sigma=100, amount=1)

    def test_direct_filtering_never_adds_noise_or_invents_reference(self):
        before = self.noisy.copy()
        with patch("spatial_denoise.core.add_noise", side_effect=AssertionError("Must not add noise")):
            result = run_pipeline(self.noisy, self.config)
        np.testing.assert_array_equal(self.noisy, before)
        np.testing.assert_array_equal(result.images["noisy"], before)
        self.assertFalse(np.shares_memory(result.images["noisy"], self.noisy))
        self.assertEqual(result.images["mean"][7, 8], 142)
        np.testing.assert_array_equal(result.images["median"], self.clean)
        self.assertNotIn("original", result.images)
        self.assertNotIn("original", result.gradients)
        for row in result.metrics:
            for key in ("mse", "psnr_db", "sobel_mse", "edge_f1"):
                self.assertIsNone(row[key])

    def test_clean_reference_supplies_metrics_and_is_copied(self):
        result = run_pipeline(self.noisy, self.config, reference=self.clean, reference_source="clean.png")
        self.assertTrue(result.has_reference)
        rows = {row["method"]: row for row in result.metrics}
        self.assertAlmostEqual(rows["noisy"]["mse"], 127**2 / self.noisy.size)
        self.assertEqual(rows["median"]["mse"], 0)
        self.assertTrue(np.isposinf(rows["median"]["psnr_db"]))
        self.assertEqual(rows["median"]["edge_f1"], 1)
        self.assertFalse(np.shares_memory(result.images["original"], self.clean))

    def test_rejects_invalid_reference_and_mode(self):
        for reference in (np.zeros((2, 2), np.uint8), np.zeros(self.noisy.shape)):
            with self.subTest(shape=reference.shape, dtype=reference.dtype):
                with self.assertRaises(ValueError):
                    run_pipeline(self.noisy, self.config, reference=reference)
        with self.assertRaises(ValueError):
            run_pipeline(self.clean, RunConfig(input_mode="bad"))
        with self.assertRaises(ValueError):
            run_pipeline(self.clean, RunConfig(), reference=self.clean)
        with self.assertRaises(ValueError):
            add_noise(self.noisy, self.config)


if __name__ == "__main__":
    unittest.main()
