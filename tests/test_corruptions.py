import unittest
import numpy as np

from src import config as C
from src import corruptions


def _images(n=20):
    return np.random.RandomState(0).randint(0, 256, (n, 64, 64, 3), dtype=np.uint8)


class CorruptionTests(unittest.TestCase):
    def test_deterministic_and_shape(self):
        x = _images()
        for name in C.CORRUPTIONS:
            a = corruptions.corrupt(x, name, 3)
            b = corruptions.corrupt(x, name, 3)
            self.assertTrue(np.array_equal(a, b), name)
            self.assertEqual(a.shape, x.shape)
            self.assertEqual(a.dtype, np.uint8)

    def test_subset_matches_full_set(self):
        x = _images()
        full = corruptions.corrupt(x, "rain", 4)
        part = corruptions.corrupt(x[5:8], "rain", 4, indices=np.arange(5, 8))
        self.assertTrue(np.array_equal(full[5:8], part))

    def test_stronger_severity_changes_more(self):
        x = _images()
        for name in C.CORRUPTIONS:
            diffs = [np.abs(corruptions.corrupt(x, name, s).astype(float) - x).mean()
                     for s in C.SEVERITIES]
            self.assertTrue(diffs[0] < diffs[-1], (name, diffs))


if __name__ == "__main__":
    unittest.main()


class WeatherCalibrationTests(unittest.TestCase):
    def test_size_and_clean_half(self):
        train_x = np.random.RandomState(1).randint(0, 256, (2000, 64, 64, 3), dtype=np.uint8)
        calib = corruptions.weather_calibration_set(train_x)
        self.assertEqual(len(calib), C.CALIB_IMAGES)
        self.assertEqual(calib.dtype, np.uint8)
        chosen = np.random.RandomState(C.SEED).choice(2000, C.CALIB_IMAGES, replace=False)
        self.assertTrue(np.array_equal(calib[:C.WCAL_CLEAN], train_x[chosen[:C.WCAL_CLEAN]]))
        self.assertFalse(np.array_equal(calib[C.WCAL_CLEAN:], train_x[chosen[C.WCAL_CLEAN:]]))
        self.assertTrue(set(C.WCAL_CORRUPTIONS).isdisjoint(C.HELDOUT_CORRUPTIONS))
