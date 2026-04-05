import numpy as np


def _validate_inputs(sampled_patchtokens, visibility_mask=None):
    sampled_patchtokens = np.asarray(sampled_patchtokens)
    if sampled_patchtokens.ndim != 3:
        raise ValueError(
            f"sampled_patchtokens must have shape [V, N, C], got {sampled_patchtokens.shape}"
        )

    n_views, n_voxels, _ = sampled_patchtokens.shape

    if visibility_mask is None:
        return sampled_patchtokens, None

    visibility_mask = np.asarray(visibility_mask)
    if visibility_mask.shape != (n_views, n_voxels):
        raise ValueError(
            f"visibility_mask must have shape {(n_views, n_voxels)}, got {visibility_mask.shape}"
        )
    return sampled_patchtokens, visibility_mask


def aggregate_mean(sampled_patchtokens):
    """Baseline aggregation: f_i = (1 / N_i) * sum_v f_iv.

    Args:
        sampled_patchtokens: np.ndarray [V, N, C]
            V views, N voxels, C feature channels.

    Returns:
        np.ndarray [N, C] float32
    """
    sampled_patchtokens, _ = _validate_inputs(sampled_patchtokens)
    return sampled_patchtokens.mean(axis=0, dtype=np.float32)


def aggregate_visible_only(sampled_patchtokens, visibility_mask, eps=1e-6):
    """Visible-only aggregation: f_i = sum_v(m_iv * f_iv) / (sum_v m_iv + eps).

    Args:
        sampled_patchtokens: np.ndarray [V, N, C]
        visibility_mask: np.ndarray [V, N], binary or boolean visibility mask.
        eps: float, numerical stability constant.

    Returns:
        np.ndarray [N, C] float32
    """
    sampled_patchtokens, visibility_mask = _validate_inputs(sampled_patchtokens, visibility_mask)
    visibility_mask = visibility_mask.astype(np.float32)

    weighted_sum = (visibility_mask[..., None] * sampled_patchtokens).sum(axis=0, dtype=np.float32)
    visibility_count = visibility_mask.sum(axis=0, dtype=np.float32)[:, None]
    return weighted_sum / (visibility_count + eps)


def aggregate_patchtokens(sampled_patchtokens, aggregation_mode='visible_only', visibility_mask=None, eps=1e-6):
    """Dispatch multiview voxel feature aggregation.

    Supported modes:
        - mean
        - visible_only (default)
    """
    if aggregation_mode == 'mean':
        return aggregate_mean(sampled_patchtokens)
    if aggregation_mode == 'visible_only':
        if visibility_mask is None:
            raise ValueError('visibility_mask is required when aggregation_mode="visible_only"')
        return aggregate_visible_only(sampled_patchtokens, visibility_mask, eps=eps)
    raise ValueError(
        f'Unsupported aggregation_mode="{aggregation_mode}". Supported: ["mean", "visible_only"]'
    )
