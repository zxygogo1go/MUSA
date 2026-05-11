import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SmokeTests(unittest.TestCase):
    def test_import_and_model_type_aliases(self):
        import musa

        self.assertTrue(hasattr(musa, "utils_dataloader"))
        self.assertTrue(hasattr(musa, "utils_warp"))
        self.assertFalse(hasattr(musa, "utils_io"))
        self.assertFalse(hasattr(musa, "warp_modules"))

        aliases = {
            "01voxelmorph-vf": "01voxelmorph-v1",
            "02resunet-vf": "02resunet-v1",
            "03lkunet-vf-lk09": "03lkunet-v1-lk09",
            "04transmorph-vf": "04transmorph-v1",
            "05dualprnet-vf": "05dualprnet-v1",
        }
        for old_name, canonical_name in aliases.items():
            self.assertEqual(musa.utils_model_zoo.normalize_model_type(old_name), canonical_name)

    def test_file_list_reader(self):
        from musa.utils_dataloader import read_file_list

        with tempfile.TemporaryDirectory() as tmpdir:
            list_path = Path(tmpdir) / "cases.txt"
            list_path.write_text("case001\n\ncase002\n", encoding="utf-8")

            self.assertEqual(read_file_list(str(list_path)), ["case001", "case002"])
            self.assertEqual(
                read_file_list(str(list_path), prefix="pre_", suffix=".npy"),
                ["pre_case001.npy", "pre_case002.npy"],
            )

    def test_identity_warp(self):
        from musa.utils_warp import SpatialTransformer

        src = torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(1, 1, 4, 5, 6)
        flow = torch.zeros(1, 3, 4, 5, 6)
        warped = SpatialTransformer((4, 5, 6))(src, flow)

        np.testing.assert_allclose(warped.detach().numpy(), src.detach().numpy(), atol=1e-5)

    def test_dataprep_onehot(self):
        import musa

        seg = torch.tensor([[[[[0, 1], [2, 0]], [[1, 2], [0, 0]]]]])
        onehot = musa.utils_dataprep.to_onehot(seg, num_classes=3)

        self.assertEqual(tuple(onehot.shape), (1, 3, 2, 2, 2))
        self.assertTrue(torch.allclose(onehot.sum(dim=1), torch.ones(1, 2, 2, 2)))


if __name__ == "__main__":
    unittest.main()
