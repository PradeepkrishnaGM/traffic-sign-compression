import unittest
import numpy as np
from src import config as C
from src import data


# these tests need the GTSRB training images, which are not part of the repository
@unittest.skipUnless(C.TRAIN_DIR.exists(), "GTSRB not downloaded")
class SplitTests(unittest.TestCase):
    def test_no_track_in_both_sets(self):
        d = data.load_arrays()
        is_val = data.split_by_track(d["train_y"], d["train_track"])
        # a track id is only unique inside a class, so compare (class, track) pairs
        pairs = d["train_y"] * 100000 + d["train_track"]
        self.assertEqual(len(set(pairs[is_val]) & set(pairs[~is_val])), 0)
        self.assertTrue(0.05 < is_val.mean() < 0.15)

    def test_split_is_deterministic(self):
        d = data.load_arrays()
        a = data.split_by_track(d["train_y"], d["train_track"])
        b = data.split_by_track(d["train_y"], d["train_track"])
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
