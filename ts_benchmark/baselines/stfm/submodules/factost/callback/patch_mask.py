import torch
from torch import nn
import math


def create_patch(xb, patch_len, stride):
    """
    Create patches from input time series with padding if necessary.

    Args:
        xb (Tensor): Input tensor of shape [batch_size, sequence_length, n_variables]
        patch_len (int): Length of each patch
        stride (int): Stride between consecutive patches

    Returns:
        (patched_tensor, num_patches)
            patched_tensor: [batch_size, num_patches, n_variables, patch_len]
            num_patches: int

    Notes:
        - Robust to short sequences (seq_len < patch_len)
        - Uses right-side replication padding to meet the exact target length
    """
    bs, seq_len, n_vars = xb.shape[0], xb.shape[1], xb.shape[2]

    # Compute intended number of patches on the original length
    num_patch = max(1, math.ceil((seq_len - patch_len) / stride) + 1)
    # Total length required to extract num_patch patches with given stride
    tgt_len = patch_len + stride * (num_patch - 1)

    # Right-pad to target length if needed (replicate last value)
    ts_pad_num = max(0, tgt_len - seq_len)
    if ts_pad_num > 0:
        ts_padding = nn.ReplicationPad1d((0, ts_pad_num))
        xb = xb.transpose(1, 2)            # [bs, n_vars, seq_len]
        xb = ts_padding(xb)                # pad on temporal dim
        xb = xb.transpose(1, 2)            # [bs, seq_len+pad, n_vars]

    # If we somehow still have insufficient length, clamp size to available
    cur_len = xb.shape[1]
    if cur_len < patch_len:
        # Force a single patch by padding up to patch_len
        extra = patch_len - cur_len
        ts_padding = nn.ReplicationPad1d((0, extra))
        xb = xb.transpose(1, 2)
        xb = ts_padding(xb)
        xb = xb.transpose(1, 2)
        cur_len = xb.shape[1]
        num_patch = 1
        tgt_len = patch_len

    # Select exactly the target window from the end and unfold into patches
    s_begin = cur_len - tgt_len
    xb = xb[:, s_begin:, :]                  # [bs, tgt_len, n_vars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)  # [bs, num_patch, n_vars, patch_len]

    return xb, num_patch


def create_patch_history_st(xb, patch_len, stride):
    """
    Create patches from spatio-temporal data with padding if necessary.
    Only applies patches to value features, keeping time features unchanged.
    
    Args:
        xb (Tensor): Input tensor of shape [batch_size × sequence_length × num_nodes × channels]
        patch_len (int): Length of each patch
        stride (int): Stride between consecutive patches
    
    Returns:
        tuple: (patched_tensor, number_of_patches)
        - patched_tensor: [batch_size × num_patches × num_nodes × channels × patch_length]
        - number_of_patches: Total number of patches created
    """
    bs, seq_len, num_nodes, channels = xb.shape
    
    # Separate value features from time features
    value_features = xb[..., 0:1]  # [bs, seq_len, num_nodes, 1]
    time_features = xb[..., 1:]    # [bs, seq_len, num_nodes, 2]
    # Process value features with patches
    value_features = value_features.reshape(bs, seq_len, num_nodes)
    
    # Calculate number of patches needed
    num_patch = math.ceil((seq_len - patch_len) / stride) + 1
    # Calculate padding
    if seq_len <= patch_len:
        ts_pad_num = patch_len - seq_len
    else:
        if seq_len % stride == 0: #不能写成 seq_len & stride == 0
            ts_pad_num = 0
        else:
            ts_pad_num = (seq_len // stride) * stride + patch_len - seq_len
    # Apply padding to value features
    ts_padding = nn.ReplicationPad1d((0, ts_pad_num))
    value_features = value_features.transpose(1, 2)
    value_features = ts_padding(value_features)
    value_features = value_features.transpose(1, 2)
    # Calculate target length and starting point
    bs, seq_len, num_nodes = value_features.shape
    tgt_len = patch_len + stride * (num_patch - 1)
    s_begin = seq_len - tgt_len

    # Extract relevant sequence and create patches for value features
    value_features = value_features[:, s_begin:, :]  # Select required sequence length
    # Create patches using unfold operation
    value_features = value_features.unfold(dimension=1, size=patch_len, step=stride)
    
    # Reshape value features back to include channel dimension
    value_features = value_features.reshape(bs, num_patch, num_nodes, 1, patch_len)
    
    # Process time features
    # Take the first value of each patch for time features
    time_features = time_features[:, ::stride, :, :]  # [bs, num_patch, num_nodes, 2]
    # Expand time features to match patch length
    time_features = time_features.unsqueeze(-1).expand(-1, -1, -1, -1, patch_len)
    
    # Combine patched value features with time features
    xb_patch = torch.cat([value_features, time_features], dim=3)
    
    return xb_patch, num_patch


def create_patch_future_st(future_features, patch_len, num_patches):
    """
    Create patches for future temporal features using non-overlapping windows.
    Matches the logic of create_patch_history_st by using the first time step of each patch.

    Args:
        future_features (Tensor): [bs, target_dim, n_vars, features]
        patch_len (int): Length of each patch
        num_patches (int): Number of patches to create

    Returns:
        Tensor: [bs, num_patches, n_vars, 1+features, patch_len]
    """
    bs, target_dim, n_vars, n_feats = future_features.shape
    
    # 1. Select starting points for each patch (non-overlapping)
    # Since future patches are contiguous and non-overlapping, stride = patch_len
    # We take the first time step of each patch window
    indices = torch.arange(0, num_patches * patch_len, patch_len, device=future_features.device)
    
    # Handle case where indices might go out of bounds (padding needed)
    # But first, let's extract what we can
    valid_indices = indices[indices < target_dim]
    
    # Extract features at start points: [bs, valid_patches, n_vars, features]
    selected_features = future_features[:, valid_indices, :, :]
    
    # If we need more patches than available time steps (padding case)
    if len(valid_indices) < num_patches:
        num_pad = num_patches - len(valid_indices)
        # Pad with the last available time step (or 0, but replication is safer)
        last_feat = future_features[:, -1:, :, :]
        padding = last_feat.repeat(1, num_pad, 1, 1)
        selected_features = torch.cat([selected_features, padding], dim=1)
        
    # 2. Expand to match patch length: [bs, num_patches, n_vars, features, patch_len]
    selected_features = selected_features.unsqueeze(-1).expand(-1, -1, -1, -1, patch_len)
    
    # 3. Add dummy value channel at the beginning to match history_data structure
    # Structure: [value(1), features...]
    dummy_val = torch.zeros(bs, num_patches, n_vars, 1, patch_len, device=future_features.device)
    
    # Combine: [bs, num_patches, n_vars, 1+features, patch_len]
    # Transpose features to match cat dimension: [bs, num_patches, n_vars, features, patch_len] -> same
    patch_combined = torch.cat([dummy_val, selected_features], dim=3)
    
    return patch_combined
