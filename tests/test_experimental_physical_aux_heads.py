import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from src.time_jepa.physical_aux_head_sweep import (
        TubeletDenseMultiScaleFusionHead,
        TubeletTemporalSelfAttentionHead,
        build_physical_head,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class ExperimentalPhysicalAuxHeadTest(unittest.TestCase):
    def test_experimental_heads_preserve_tubelet_layout(self):
        specs = (
            ("temporal_multiscale_fusion", 18),
            ("temporal_attention_local", 16),
            ("temporal_attention_global", 16),
            ("temporal_attention", 16),
        )
        for head_type, hidden_dim in specs:
            head = build_physical_head(
                head_type=head_type,
                embed_dim=8,
                num_outputs=3,
                num_frames=12,
                tubelet_size=2,
                hidden_dim=hidden_dim,
            )
            features = torch.randn(2, 6 * 4, 8, requires_grad=True)
            predictions = head(features)
            self.assertEqual(tuple(predictions.shape), (2, 6, 3))
            self.assertTrue(bool(torch.isfinite(predictions).all()))
            predictions.square().mean().backward()
            self.assertIsNotNone(features.grad)
            self.assertTrue(bool(torch.isfinite(features.grad).all()))

    def test_h3_strong_has_dense_branches_and_concat_fusion(self):
        head = TubeletDenseMultiScaleFusionHead(
            embed_dim=32,
            num_outputs=9,
            num_frames=64,
            tubelet_size=2,
            hidden_dim=20,
        )
        self.assertEqual(
            tuple(branch.kernel_size[0] for branch in head.temporal_branches),
            (3, 5, 9),
        )
        self.assertEqual(head.fusion.in_channels, 60)
        self.assertEqual(head.fusion.out_channels, 20)

    def test_attention_masks_match_local_global_and_hybrid_definitions(self):
        def make(mode):
            return TubeletTemporalSelfAttentionHead(
                embed_dim=32,
                num_outputs=9,
                num_frames=64,
                tubelet_size=2,
                hidden_dim=416,
                num_heads=8,
                local_window_size=9,
                attention_mode=mode,
            )

        local = make("local")
        self.assertEqual(tuple(local._attention_mask.shape), (32, 32))
        self.assertFalse(bool(local._attention_mask[10, 6]))
        self.assertFalse(bool(local._attention_mask[10, 14]))
        self.assertTrue(bool(local._attention_mask[10, 5]))
        self.assertTrue(bool(local._attention_mask[10, 15]))

        global_head = make("global")
        self.assertEqual(global_head._attention_mask.numel(), 0)
        self.assertIsNone(global_head._expanded_attention_mask(batch_size=2))

        hybrid = make("hybrid")
        self.assertEqual(tuple(hybrid._attention_mask.shape), (8, 32, 32))
        self.assertTrue(torch.equal(hybrid._attention_mask[0], local._attention_mask))
        self.assertTrue(torch.equal(hybrid._attention_mask[3], local._attention_mask))
        self.assertFalse(bool(hybrid._attention_mask[4:].any()))
        self.assertEqual(
            tuple(hybrid._expanded_attention_mask(batch_size=2).shape),
            (16, 32, 32),
        )

    def test_egoexo_parameter_budgets_match_the_design(self):
        specs = {
            "conv1x1": (256, 1_996_553),
            "temporal_bottleneck_k1": (1408, 2_002_185),
            "temporal_bottleneck_k3": (1408, 2_005_001),
            "temporal_multiscale": (1408, 2_016_265),
            "temporal_multiscale_fusion": (610, 1_996_915),
            "temporal_attention_local": (416, 1_995_049),
            "temporal_attention_global": (416, 1_995_049),
            "temporal_attention": (416, 1_995_049),
        }
        for head_type, (hidden_dim, expected_count) in specs.items():
            head = build_physical_head(
                head_type=head_type,
                embed_dim=1408,
                num_outputs=9,
                num_frames=64,
                tubelet_size=2,
                hidden_dim=hidden_dim,
                temporal_dilations=(1, 2, 4),
            )
            actual_count = sum(parameter.numel() for parameter in head.parameters())
            self.assertEqual(
                actual_count,
                expected_count,
                msg=head_type,
            )


if __name__ == "__main__":
    unittest.main()
