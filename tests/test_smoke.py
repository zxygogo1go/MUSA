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

    def test_musa_plus_masks_roi_and_difficulty(self):
        import musa

        seg = torch.tensor([[[[[0, 1, 0], [2, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 3, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]])
        mask = musa.utils_musa_plus.seg_to_label_mask(seg, labels=[1, 3])
        self.assertEqual(tuple(mask.shape), (1, 1, 3, 3, 3))
        self.assertEqual(float(mask.sum()), 2.0)

        gate = musa.utils_musa_plus.build_roi_gate(mask, radius=1, smooth_steps=1)
        self.assertEqual(tuple(gate.shape), tuple(mask.shape))
        self.assertGreater(float(gate.sum()), float(mask.sum()))
        self.assertTrue(torch.all(gate >= 0))
        self.assertTrue(torch.all(gate <= 1))

        batch_mask = torch.cat((mask, mask), dim=0)
        batch_gate = musa.utils_musa_plus.build_roi_gate_per_batch(batch_mask, torch.tensor([0, 1]), smooth_steps=0)
        self.assertEqual(tuple(batch_gate.shape), tuple(batch_mask.shape))
        self.assertAlmostEqual(float(batch_gate[0].sum()), float(mask.sum()))
        self.assertGreater(float(batch_gate[1].sum()), float(mask.sum()))

        moving = torch.zeros(1, 1, 3, 3, 3)
        fixed = moving.clone()
        difficulty = musa.utils_musa_plus.estimate_pair_difficulty(
            moving,
            fixed,
            moving_oar_mask=mask,
            fixed_oar_mask=mask,
            moving_bone_mask=mask,
            fixed_bone_mask=mask,
        )
        self.assertTrue(torch.allclose(difficulty, torch.zeros_like(difficulty)))

        stage2_difficulty = musa.utils_musa_plus.estimate_stage2_pair_difficulty(
            fixed=fixed,
            deformed_stage2=fixed,
            dvf_stage2=torch.zeros(1, 3, 3, 3, 3),
            warped_small_mask_stage2=mask,
            fixed_small_mask=mask,
            warped_bone_mask_stage2=mask,
            fixed_bone_mask=mask,
            image_mask=mask,
        )
        self.assertTrue(torch.allclose(stage2_difficulty, torch.zeros_like(stage2_difficulty)))

        loss_per_batch = musa.utils_musa_plus.masked_mse_loss_per_batch(
            pred=torch.zeros(2, 1, 3, 3, 3),
            target=torch.ones(2, 1, 3, 3, 3),
            mask=batch_mask,
        )
        self.assertEqual(tuple(loss_per_batch.shape), (2,))

        jac_loss = musa.utils_musa_plus.jacobian_hinge_loss_per_batch(
            torch.zeros(2, 3, 3, 3, 3),
            roi_gate=batch_mask,
        )
        self.assertEqual(tuple(jac_loss.shape), (2,))
        self.assertTrue(torch.allclose(jac_loss, torch.zeros_like(jac_loss)))

    def test_musa_plus_local_residual_unet_shape(self):
        from musa.registration_models.musa_plus import LocalResidualUNet

        model = LocalResidualUNet(in_channels=7, filters=(2, 4, 8), instance_norm=False)
        x = torch.randn(1, 7, 8, 8, 8)
        y = model(x)

        self.assertEqual(tuple(y.shape), (1, 3, 8, 8, 8))


if __name__ == "__main__":
    unittest.main()
