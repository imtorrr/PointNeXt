"""
PyTorch Geometric compatible test function for variable-sized point clouds.

This is a simplified test function designed for:
-
- Variable-sized point clouds (variable=True in dataset config)
- Full-tile inference without complex multi-voxel voting

Key differences from original test():
- Processes full tiles directly (no sub-cloud splitting)
- Uses PyG's native batching (Data objects with batch attribute)
- No reshape operations for logits
- Simpler, more straightforward inference pipeline
"""

from openpoints.utils.random import set_random_seed
import torch
import numpy as np
import logging
import pickle
import os
from tqdm import tqdm
from torch_geometric.data import Data
from openpoints.utils import ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys


@torch.no_grad()
def test_pyg_variable(model, test_loader, cfg, num_votes=1):
    """Test on full point clouds with PyG variable-sized data using DataLoader.

    This function is optimized for PyTorch Geometric models that handle variable-sized inputs.
    It uses the dataset's built-in DataLoader for consistent data handling.

    Args:
        model: PyG-compatible model (e.g., VariableSeg)
        test_loader: DataLoader for test set
        cfg: Configuration object
        num_votes: Number of voting rounds (default: 1, currently only 1 is supported)

    Returns:
        miou, macc, oa, ious, accs, confusion_matrix
    """
    model.eval()
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    set_random_seed(0)
    cfg.visualize = cfg.get("visualize", False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj

        cfg.vis_dir = os.path.join(cfg.run_dir, "visualization")
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.0

    dataset_name = cfg.dataset.common.NAME.lower()
    cfg.save_path = cfg.get(
        "save_path",
        f"results/{cfg.task_name}/{cfg.dataset.test.split}/{cfg.cfg_basename}",
    )
    os.makedirs(cfg.save_path, exist_ok=True)

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
    for cloud_idx, data in pbar:
        # Move batch to GPU
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)

        # Extract label and features
        label = data.get("y", data.get("label", None))
        # if label is not None:
        #     label = label.squeeze(-1)
        #     # Apply label offset if configured
        #     if cfg.dataset.common.get("label_offset", 0) != 0:
        #         label = label + cfg.dataset.common.label_offset

        # Extract features
        data["x"] = get_features_by_keys(data, cfg.feature_keys)

        # Convert to PyG Data format if not already
        if not isinstance(data, Data):
            data = Data(**data)

        # Forward pass
        logits = model(data)  # Shape: (num_points, num_classes)

        # Get predictions
        pred = logits.argmax(dim=1)

        # Compute metrics if labels available
        if label is not None:
            all_cm.update(pred, label)

        # Visualization
        if cfg.visualize:
            coord = data.pos.cpu().numpy().copy()

            # Apply batch offset for visualization (spread clouds in a grid)
            if hasattr(data, "batch") and data.batch is not None:
                batch_indices = data.batch.cpu().numpy()
                unique_batches = np.unique(batch_indices)

                # Calculate grid layout (cols x rows)
                num_batches = len(unique_batches)
                grid_cols = int(np.ceil(np.sqrt(num_batches)))

                # Apply grid offset (8 meters spacing)
                for batch_id in unique_batches:
                    mask = batch_indices == batch_id
                    grid_row = batch_id // grid_cols
                    grid_col = batch_id % grid_cols

                    coord[mask, 0] += grid_col * 8  # x offset
                    coord[mask, 1] += grid_row * 8  # y offset

            gt = label.cpu().numpy().squeeze() if label is not None else None
            pred_vis = pred.cpu().numpy().squeeze()
            gt = cfg.cmap[gt, :] if gt is not None else None
            pred_vis = cfg.cmap[pred_vis, :]

            file_name = f"{dataset_name}-{cloud_idx}"

            # output ground truth labels
            if gt is not None:
                write_obj(coord, gt, os.path.join(cfg.vis_dir, f"gt-{file_name}.pcd"))
            # output pred labels
            write_obj(
                coord,
                pred_vis,
                os.path.join(cfg.vis_dir, f"{cfg.cfg_basename}-{file_name}.pcd"),
            )

    # Calculate final metrics
    tp, union, count = all_cm.tp, all_cm.union, all_cm.count
    if cfg.distributed:
        import torch.distributed as dist

        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)

    miou, macc, oa, ious, accs = get_mious(tp, union, count)

    # Save confusion matrix for later analysis
    cm_save = {
        "tp": tp,
        "union": union,
        "count": count,
    }
    cm_path = os.path.join(cfg.run_dir, "confusion_matrix.pkl")
    with open(cm_path, "wb") as f:
        pickle.dump(cm_save, f)
    logging.info(f"Confusion matrix saved to {cm_path}")

    return miou, macc, oa, ious, accs, all_cm
