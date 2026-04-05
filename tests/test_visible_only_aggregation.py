"""Unit tests for multiview voxel feature aggregation.

Assumed tensor dimensions (inferred from extract_feature pipeline):
- V: number of views
- N: number of voxels
- C: feature channels
- sampled_patchtokens has shape [V, N, C]
- visibility_mask has shape [V, N]
"""

import unittest

import numpy as np

from dataset_toolkits.aggregation_utils import (
    aggregate_mean,
    aggregate_patchtokens,
    aggregate_visible_only,
)


class TestVisibleOnlyAggregation(unittest.TestCase):
    def test_shape_inference_and_output_shape(self):
        sampled = np.zeros((3, 4, 2), dtype=np.float32)
        mask = np.ones((3, 4), dtype=np.float32)

        out_mean = aggregate_patchtokens(sampled, aggregation_mode='mean')
        out_visible = aggregate_patchtokens(sampled, aggregation_mode='visible_only', visibility_mask=mask)

        self.assertEqual(out_mean.shape, (4, 2))
        self.assertEqual(out_visible.shape, (4, 2))

    def test_baseline_mean_matches_manual_example(self):
        # Hand-computed example with V=3, N=2, C=2.
        sampled = np.array(
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
                [[5.0, 50.0], [6.0, 60.0]],
            ],
            dtype=np.float32,
        )
        expected = np.array(
            [[3.0, 30.0], [4.0, 40.0]],
            dtype=np.float32,
        )

        actual = aggregate_mean(sampled)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_visible_only_matches_manual_example(self):
        # m_iv selects only visible views per voxel.
        sampled = np.array(
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
                [[5.0, 50.0], [6.0, 60.0]],
            ],
            dtype=np.float32,
        )
        mask = np.array(
            [
                [1, 0],
                [1, 1],
                [0, 1],
            ],
            dtype=np.float32,
        )
        eps = 1e-6
        # Follow formula exactly: sum(m_iv * f_iv) / (sum(m_iv) + eps)
        expected = (sampled * mask[..., None]).sum(axis=0) / (mask.sum(axis=0, keepdims=True).T + eps)

        actual = aggregate_visible_only(sampled, mask, eps=eps)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_visible_only_zero_mask_uses_eps_denominator(self):
        sampled = np.array(
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
            ],
            dtype=np.float32,
        )
        mask = np.zeros((2, 1), dtype=np.float32)

        actual = aggregate_visible_only(sampled, mask, eps=1e-6)
        expected = np.array([[0.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_invalid_mask_shape_raises(self):
        sampled = np.zeros((2, 3, 4), dtype=np.float32)
        bad_mask = np.zeros((2, 4), dtype=np.float32)

        with self.assertRaises(ValueError):
            aggregate_patchtokens(
                sampled,
                aggregation_mode='visible_only',
                visibility_mask=bad_mask,
            )


if __name__ == '__main__':
    unittest.main()
