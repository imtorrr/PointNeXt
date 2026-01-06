"""
PyTorch Geometric compatible test function for variable-sized point clouds.

This is a simplified test function designed for:
- PyTorch Geometric models (VariableSeg, etc.)
- Variable-sized point clouds (variable=True in dataset config)
- Full-tile inference without complex multi-voxel voting

Key differences from original test():
- Processes full tiles directly (no sub-cloud splitting)
- Uses PyG's native batching (Data objects with batch attribute)
- No reshape operations for logits
- Simpler, more straightforward inference pipeline
"""

import torch
import numpy as np
import logging
from tqdm import tqdm
from torch_geometric.data import Data
from openpoints.utils import ConfusionMatrix, get_mious
from openpoints.transforms import build_transforms_from_cfg
from openpoints.dataset import get_features_by_keys
from openpoints.dataset.data_util import voxelize


@torch.no_grad()
def test_pyg_variable(model, data_list, cfg, num_votes=1):
    """Test on full point clouds with PyG variable-sized data.

    This function is optimized for PyTorch Geometric models that handle variable-sized inputs.
    It processes entire tiles without splitting into sub-clouds.

    Args:
        model: PyG-compatible model (e.g., VariableSeg)
        data_list: List of paths to test tiles (.npy files)
        cfg: Configuration object
        num_votes: Number of voting rounds (default: 1, currently only 1 is supported)

    Returns:
        miou, macc, oa, ious, accs, confusion_matrix
    """
    model.eval()
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)

    # Build transforms
    trans_split = "val" if cfg.datatransforms.get("test", None) is None else "test"
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)

    gravity_dim = cfg.datatransforms.kwargs.gravity_dim
    voxel_size = cfg.dataset.common.get("voxel_size", None)

    for cloud_idx, data_path in enumerate(tqdm(data_list, desc="Testing")):
        logging.info(f"Test [{cloud_idx}/{len(data_list)}] cloud: {data_path}")

        # Load full tile
        coord, feat, label = load_tile_data(data_path, cfg)

        # Optional voxel downsampling for test
        if voxel_size is not None:
            idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
            # Use voxel centers (one point per voxel)
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
            idx_unique = idx_sort[idx_select]
            coord = coord[idx_unique]
            feat = feat[idx_unique] if feat is not None else None
            if label is not None:
                # Store original labels for full resolution
                label_voxel = label[idx_unique]
                # We'll use inverse mapping later if needed
                inverse_map = np.zeros(len(idx_sort), dtype=np.int64)
                for i, idx in enumerate(idx_sort):
                    inverse_map[idx] = voxel_idx[i]

        # Normalize coordinates
        coord = coord - coord.min(0)

        # Prepare PyG Data object
        data = {"pos": torch.from_numpy(coord.astype(np.float32))}
        from copy import deepcopy
        data["x"] = deepcopy(data["pos"])
        if feat is not None:
            data["x"] = torch.from_numpy(feat.astype(np.float32))

        # Apply transforms
        if pipe_transform is not None:
            data = pipe_transform(data)

        # Add heights if needed
        if "heights" in cfg.feature_keys and "heights" not in data.keys():
            data["heights"] = torch.from_numpy(
                coord[:, gravity_dim:gravity_dim + 1].astype(np.float32)
            )

        # Create batch attribute for PyG (single sample, so all points belong to batch 0)
        num_points = len(coord)
        data["batch"] = torch.zeros(num_points, dtype=torch.long)

        # Move to GPU
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)

        # Convert to PyG Data format
        pyg_data = Data(**data)

        # Forward pass
        logits = model(pyg_data)  # Shape: (num_points, num_classes)

        # Get predictions
        pred = logits.argmax(dim=1)

        # Compute metrics if labels available
        if label is not None:
            # Apply label offset if configured
            label_torch = torch.from_numpy(label.astype(np.long).squeeze()).cuda(non_blocking=True)
            if cfg.dataset.common.get("label_offset", 0) != 0:
                label_torch = label_torch + cfg.dataset.common.label_offset

            # If we downsampled, map predictions back to original resolution
            if voxel_size is not None:
                # Use nearest neighbor to assign predictions to all original points
                pred_full = pred[inverse_map]
                all_cm.update(pred_full, label_torch)
            else:
                all_cm.update(pred, label_torch)

    # Calculate final metrics
    tp, union, count = all_cm.tp, all_cm.union, all_cm.count
    if cfg.distributed:
        import torch.distributed as dist
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)

    miou, macc, oa, ious, accs = get_mious(tp, union, count)

    return miou, macc, oa, ious, accs, all_cm.confusion_matrix


def load_tile_data(data_path, cfg):
    """Load a single tile from .npy file.

    Args:
        data_path: Path to .npy file
        cfg: Configuration object

    Returns:
        coord, feat, label (numpy arrays)
    """
    data = np.load(data_path).astype(np.float32)
    coord = data[:, :3]
    feat = None
    label = None

    # Parse based on dataset configuration
    if data.shape[1] > 3:
        if cfg.dataset.common.get("label_field", None) is not None:
            # Last column is label
            if data.shape[1] > 4:
                feat = data[:, 3:-1]  # Features between XYZ and label
            label = data[:, -1]
        else:
            # No labels, all columns after XYZ are features
            feat = data[:, 3:]

    return coord, feat, label
