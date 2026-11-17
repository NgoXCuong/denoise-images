import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_denoise.core import RunConfig, run_pipeline
from spatial_denoise.imaging import load_image


def temporary_workspace():
    # Keep all test writes/cleanup inside this repository, including sandboxed runs.
    return tempfile.TemporaryDirectory(prefix=".test_", dir=Path(__file__).resolve().parents[1])


class ColorImageTests(unittest.TestCase):
    def test_color_survives_load_and_pipeline(self):
        image = np.full((5, 7, 3), [200, 80, 30], np.uint8)
        with temporary_workspace() as folder:
            source = Path(folder) / "color.png"
            Image.fromarray(image).save(source)
            loaded = load_image(source)
            np.testing.assert_array_equal(loaded, image)
            result = run_pipeline(loaded, RunConfig(sigma=0))
            for array in result.images.values():
                np.testing.assert_array_equal(array, image)


class ImageLoadingTests(unittest.TestCase):
    def test_rgba_white_background_and_rgb_preserved(self):
        with temporary_workspace() as folder:
            file = Path(folder) / "trong suốt.png"
            Image.new("RGBA", (3, 3), (0, 0, 0, 0)).save(file)
            np.testing.assert_array_equal(load_image(file), np.full((3, 3, 3), 255, np.uint8))
            Image.new("RGB", (3, 3), (255, 0, 0)).save(file)
            np.testing.assert_array_equal(load_image(file), np.full((3, 3, 3), [255, 0, 0], np.uint8))

    def test_rejects_16bit_image(self):
        with temporary_workspace() as folder:
            path = Path(folder) / "hdr.tiff"
            Image.fromarray(np.zeros((5, 5), dtype=np.uint16)).save(path)
            with self.assertRaises(ValueError):
                load_image(path)


if __name__ == "__main__":
    unittest.main()
