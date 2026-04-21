"""
Inference script for PointNeXt semantic segmentation on LAS/LAZ point clouds.

Mirrors the structure of the FSCT SemanticSegmentation pipeline, adapted for
the PointNeXt / PyTorch Geometric (VariableSeg) model used in this thesis.

Data preparation per tile exactly matches the test-time pipeline:
    [PointsToTensor, PointCloudXYZAlign]

PointCloudXYZAlign transform (applied per tile):
    pos -= mean(pos)            # centre all 3 axes
    pos[:, Z] -= min(pos[:, Z]) # set Z-minimum to zero

Coordinate restoration: orig = trans + local_shift
    local_shift = [mean_x, mean_y, min_z_raw]
    (because z_orig = z_trans + mean_z + min_z_centred = z_trans + min_z_raw)

Usage:
    python examples/segmentation/inference.py \\
        --cfg  cfgs/runpod-4090/forinstancev2/pointnext-s-pyg.yaml \\
        --input        path/to/cloud.las \\
        --checkpoint   weights/best.pth

Optional flags:
    --output        override output .las path (default: <input>.segmented.las)
    --batch_size    tile batches per forward pass (default 4)
    --cpu           force CPU
    --keep_tiles    keep intermediate tile .npy files
    --tile_size     tile edge in metres (overrides cfg)
    --tile_overlap  overlap ratio 0-1   (overrides cfg)
    --label_offset  integer added to predicted class IDs before saving
                    (e.g. 1 when the LAS file uses 1-indexed classes)
    --knn_k         neighbours for label propagation (default 16)
"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath("."))

import argparse
import glob
import logging
import shutil
import time

import laspy
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from openpoints.dataset import get_features_by_keys
from openpoints.dataset.data_util import compute_hag, tile_pc_fast
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig, load_checkpoint


# ─── Tile Dataset ─────────────────────────────────────────────────────────────


class TileDataset(Dataset):
    """Load pre-tiled .npy files and return PyG Data objects.

    Each tile has its coordinates transformed exactly like the test pipeline:
        PointsToTensor + PointCloudXYZAlign (gravity_dim=2 by default)

    The per-tile offset required to undo the transform is stored in
    ``data.local_shift`` (shape 3) so that global xyz can be recovered after
    inference as:
        xyz_global = xyz_transformed + local_shift

    If the .npy tiles contain extra feature columns (HAG, RGB, intensity…)
    they are stored in ``data.x`` so that ``get_features_by_keys`` can combine
    them with ``data.pos`` according to the config's ``feature_keys``.

    Heights (Z after PointCloudXYZAlign) are stored in ``data.heights`` to
    support ``feature_keys`` values such as ``"pos,heights"``.
    """

    def __init__(self, tile_dir: str, gravity_dim: int = 2):
        self.filenames = sorted(glob.glob(os.path.join(tile_dir, "*.npy")))
        self.gravity_dim = gravity_dim
        if not self.filenames:
            raise ValueError(f"No .npy tiles found in {tile_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        pc = np.load(self.filenames[index]).astype(np.float32)
        coord = pc[:, :3].copy()  # (N, 3)  raw xyz
        feat = pc[:, 3:] if pc.shape[1] > 3 else None  # (N, C) or None

        # ── PointCloudXYZAlign ──────────────────────────────────────────────
        # Step 1: centre all three axes around the tile mean
        mean_pos = coord.mean(axis=0)  # [mx, my, mz]
        coord -= mean_pos

        # Step 2: set Z-minimum to zero (gravity_dim = 2)
        min_z_centred = coord[:, self.gravity_dim].min()
        coord[:, self.gravity_dim] -= min_z_centred

        # Shift that maps transformed → original
        #   x_orig = x_trans + mean_x
        #   y_orig = y_trans + mean_y
        #   z_orig = z_trans + mean_z + min_z_centred  = z_trans + min_z_raw
        local_shift = mean_pos.copy()
        local_shift[self.gravity_dim] += min_z_centred  # = min_z_raw

        # ── Assemble PyG Data ───────────────────────────────────────────────
        pos = torch.from_numpy(coord)  # (N, 3)
        data = Data(pos=pos)
        data.local_shift = torch.from_numpy(local_shift)  # (3,)

        # Heights: Z coordinate after PointCloudXYZAlign (matches LASDataset)
        data.heights = pos[:, self.gravity_dim : self.gravity_dim + 1]  # (N, 1)

        if feat is not None:
            data.x = torch.from_numpy(feat)  # (N, C)

        return data


def collate_pyg(data_list):
    """Collate a list of Data objects into a PyG Batch."""
    return Batch.from_data_list(data_list)


# ─── Label propagation ────────────────────────────────────────────────────────


def _process_propagation_chunk(
    chunk_idx: int,
    original_cloud: np.ndarray,
    probs_all: np.ndarray,
    nn: NearestNeighbors,
    chunk_size: int,
    k: int,
    direct_threshold: float,
) -> tuple:
    """Process a single chunk for label propagation.

    Args:
        chunk_idx:        Index of the chunk.
        original_cloud:   (M, 3+) full-resolution global xyz.
        probs_all:        (N, num_classes) soft class probabilities from model.
        nn:               Fitted NearestNeighbors object.
        chunk_size:       Number of points per chunk.
        k:                Number of neighbours for the vote.
        direct_threshold: Distance threshold for direct inference.

    Returns:
        Tuple of (start_idx, labels_chunk, model_inferred_chunk)
    """
    M = len(original_cloud)
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, M)

    distances, indices = nn.kneighbors(original_cloud[start:end, :3])  # (C, k)
    avg_probs = np.mean(probs_all[indices], axis=1)  # (C, num_classes)
    labels_chunk = np.argmax(avg_probs, axis=1)
    model_inferred_chunk = (distances[:, 0] < direct_threshold).astype(np.uint8)

    return start, end, labels_chunk, model_inferred_chunk


def propagate_labels(
    pred_cloud: np.ndarray,
    original_cloud: np.ndarray,
    k: int = 16,
    direct_threshold: float = 0.05,
    chunk_size: int = 1_000_000,
    n_jobs: int = -1,
) -> tuple:
    """Assign class labels from a (possibly downsampled) segmented cloud to
    every point in the original full-resolution cloud via k-NN voting.

    Queries are processed in ``chunk_size`` chunks in parallel to avoid allocating
    (M × k) distance/index arrays for very large clouds (e.g. 200 M pts).

    Args:
        pred_cloud:        (N, 3 + num_classes)  global xyz + soft class probs.
        original_cloud:    (M, 3+)               full-resolution global xyz.
        k:                 Number of neighbours for the vote.
        direct_threshold:  Distance (metres) within which a point is considered
                           directly inferred by the model rather than propagated.
        chunk_size:        Number of original-cloud points to query per batch.
        n_jobs:            Number of parallel workers (-1 = all cores, default -1).

    Returns:
        labels:          argmax label per point, shape (M,), dtype float64.
        model_inferred:  uint8 mask (M,) — 1 if directly inferred, 0 if k-NN propagated.
    """
    print("Propagating labels to original cloud…")
    M = len(original_cloud)
    num_classes = pred_cloud.shape[1] - 3
    probs_all = pred_cloud[:, 3:]  # (N, num_classes)

    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", metric="euclidean")
    nn.fit(pred_cloud[:, :3])

    labels = np.empty(M, dtype=np.float64)
    model_inferred = np.empty(M, dtype=np.uint8)

    n_chunks = (M + chunk_size - 1) // chunk_size

    # Process chunks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_propagation_chunk)(
            i,
            original_cloud,
            probs_all,
            nn,
            chunk_size,
            k,
            direct_threshold,
        )
        for i in tqdm(range(n_chunks), desc="Propagating", unit="chunk")
    )

    # Aggregate results
    for start, end, labels_chunk, model_inferred_chunk in results:
        labels[start:end] = labels_chunk
        model_inferred[start:end] = model_inferred_chunk

    n_direct = model_inferred.sum()
    print(
        f"  {n_direct:,} / {M:,} points directly inferred ({100 * n_direct / M:.1f}%)"
    )

    return labels, model_inferred


# ─── Main inference class ──────────────────────────────────────────────────────


class SemanticSegmentation:
    """End-to-end inference pipeline for PointNeXt segmentation on LAS/LAZ files.

    Parameters
    ----------
    parameters : dict
        Required keys
        -------------
        point_cloud_filename  Path to the input LAS/LAZ file.
        checkpoint_path       Path to the model checkpoint (.pth).
        cfg_path              Path to the YAML model/dataset config.

        Optional keys
        -------------
        batch_size            Tile batches per forward pass (default 4).
        use_cpu               Force CPU (default False).
        delete_working_dir    Remove tile cache after inference (default True).
        tile_size             Tile edge length in metres (default: from cfg or 6.0).
        tile_overlap          Tile overlap ratio 0-1 (default: from cfg or 0.5).
        min_points_per_tile   Minimum points to retain a tile (default 2000).
        label_offset          Integer added to predicted class IDs before saving
                              (e.g. 1 when the LAS file uses 1-indexed classes).
        knn_k                 Neighbours for label propagation (default 16).
        use_existing_tiles    Use existing tiles if available; generate if missing (default False).
    """

    def __init__(self, parameters: dict):
        self.start_time = time.time()
        self.params = parameters

        # ── Device ──────────────────────────────────────────────────────────
        use_cpu = self.params.get("use_cpu", False)
        print("Is CUDA available?", torch.cuda.is_available())
        self.device = torch.device(
            "cuda" if (not use_cpu and torch.cuda.is_available()) else "cpu"
        )
        print("Performing inference on device:", self.device)
        if self.device.type == "cpu":
            print(
                "Please be aware that inference will be much slower on CPU. "
                "An Nvidia GPU is highly recommended."
            )

        # ── Paths ────────────────────────────────────────────────────────────
        filename = parameters["point_cloud_filename"].replace("\\", "/")
        self.input_path = filename
        base_dir = os.path.dirname(os.path.realpath(filename))
        stem = os.path.splitext(os.path.basename(filename))[0]
        self.output_las = os.path.join(base_dir, f"{stem}.segmented.las")
        self.working_dir = os.path.join(base_dir, f"{stem}_working_tiles")
        os.makedirs(self.working_dir, exist_ok=True)

        # ── Config ───────────────────────────────────────────────────────────
        self.cfg = EasyConfig()
        self.cfg.load(parameters["cfg_path"], recursive=True)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _tiles_exist(self) -> bool:
        """Check if tiles already exist in working directory."""
        return len(glob.glob(os.path.join(self.working_dir, "*.npy"))) > 0

    def _load_las(self) -> np.ndarray:
        """Load the input LAS/LAZ file with all configured features.

        Mirrors ``LASDataset._tile_las_file`` so that the feature columns in
        the tiles are identical to those produced during training.

        Feature columns appended after xyz (in order, if enabled in cfg):
            - HAG       (1 col)  use_approximate_hag
            - RGB       (3 cols) use_rgb
            - Intensity (1 col)  use_intensity
            - Return number + num_returns (2 cols) use_return_number

        Returns
        -------
        pc : np.ndarray  shape (N, 3 + num_features), float32
        Also sets ``self.original_cloud`` (N, 3), float64, for label propagation.
        """
        print(f"Loading: {self.input_path}")
        las = laspy.read(self.input_path)

        coords = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

        # Global shift: subtract the coordinate minimum so that float32
        # precision is not wasted on large UTM-scale offsets.  The shift is
        # restored in ``_save`` before writing the output LAS file.
        self.global_shift = coords.min(axis=0)  # (3,) float64
        coords = coords - self.global_shift
        print(f"  Global shift applied: {self.global_shift}")

        self.original_cloud = coords  # kept for label propagation

        # ── Read cfg feature flags ───────────────────────────────────────────
        ds_cfg = self.cfg.dataset.common
        use_hag = ds_cfg.get("use_approximate_hag", False)
        use_rgb = ds_cfg.get("use_rgb", False)
        use_intensity = ds_cfg.get("use_intensity", False)
        use_return_number = ds_cfg.get("use_return_number", False)

        features = []

        if use_hag:
            print("  Computing approximate HAG…")
            hag = compute_hag(coords.astype(np.float32), grid_size=5).reshape(-1, 1)
            features.append(hag.astype(np.float32))

        if use_rgb:
            try:
                rgb = np.vstack([las.red, las.green, las.blue]).T
                rgb = rgb / 65535.0 if rgb.max() > 255 else rgb / 255.0
                features.append(rgb.astype(np.float32))
            except AttributeError:
                logging.warning("RGB not found in LAS file – skipping.")

        if use_intensity:
            try:
                intensity = las.intensity.reshape(-1, 1).astype(np.float32)
                max_int = intensity.max()
                if max_int > 0:
                    intensity /= max_int
                features.append(intensity)
            except AttributeError:
                logging.warning("Intensity not found in LAS file – skipping.")

        if use_return_number:
            try:
                return_num = las.return_number.reshape(-1, 1).astype(np.float32)
                num_returns = las.number_of_returns.reshape(-1, 1).astype(np.float32)
                features.append(return_num)
                features.append(num_returns)
            except AttributeError:
                logging.warning("Return number not found in LAS file – skipping.")

        xyz_f32 = coords.astype(np.float32)
        pc = np.hstack([xyz_f32] + features) if features else xyz_f32
        print(
            f"  Loaded {len(coords):,} points — "
            f"3 xyz + {pc.shape[1] - 3} extra feature(s)"
        )
        return pc  # (N, 3 + C)

    def _tile(self, pc: np.ndarray):
        """Tile the full point cloud and write each tile as a .npy file.

        Uses the same ``tile_pc_fast`` function as ``LASDataset._tile_las_file``
        so the tile layout is identical to training/validation.
        """
        tile_size = self.params.get(
            "tile_size", self.cfg.dataset.common.get("tile_size", 6.0)
        )
        tile_overlap = self.params.get(
            "tile_overlap", self.cfg.dataset.common.get("tile_overlap", 0.5)
        )
        min_pts = 0
        voxel_max = self.params.get("voxel_max", None)

        print(
            f"Tiling (tile_size={tile_size} m, "
            f"overlap={tile_overlap}, min_pts={min_pts})…"
        )
        tile_pc_fast(
            pc,
            las_path=self.input_path,  # used only for naming the tile files
            output_dir=self.working_dir,
            box_dim=tile_size,
            box_overlap=tile_overlap,
            voxel_max=voxel_max,
            min_points_per_tile=min_pts,
        )
        n_tiles = len(glob.glob(os.path.join(self.working_dir, "*.npy")))
        print(f"Created {n_tiles} tiles in {self.working_dir}")

    def _build_model(self):
        """Instantiate the PointNeXt model and load the checkpoint."""
        if self.cfg.model.get("in_channels", None) is None:
            self.cfg.model.in_channels = self.cfg.model.encoder_args.in_channels
        model = build_model_from_cfg(self.cfg.model).to(self.device)
        load_checkpoint(model, pretrained_path=self.params["checkpoint_path"])
        model.eval()
        return model

    def _save(self):
        """Write the labelled point cloud as a LAS file alongside the input."""
        label_offset = self.params.get("label_offset", 0)
        xyz = self.original_cloud[:, :3] + self.global_shift  # restore global shift
        labels = (self.output_labels + label_offset).astype(np.uint8)

        header = laspy.LasHeader(point_format=0, version="1.4")
        header.offsets = xyz.min(axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])
        header.add_extra_dim(
            laspy.ExtraBytesParams(
                name="model_inferred",
                type=np.uint8,
                description="1=inferred, 0=propagated",
            )
        )
        las_out = laspy.LasData(header=header)
        las_out.x = xyz[:, 0]
        las_out.y = xyz[:, 1]
        las_out.z = xyz[:, 2]
        las_out.classification = labels
        las_out.model_inferred = self.model_inferred

        las_out.write(self.output_las)
        print(f"Saved: {self.output_las}")

    # ── Public entry point ─────────────────────────────────────────────────────

    def inference(self):
        """Run the full inference pipeline and return the labelled point cloud."""

        # 1. Load raw cloud (+ features matching training config)
        pc = self._load_las()

        # 2. Tile into small boxes saved as .npy (or reuse existing)
        use_existing = self.params.get("use_existing_tiles", False)
        if use_existing and self._tiles_exist():
            n_tiles = len(glob.glob(os.path.join(self.working_dir, "*.npy")))
            print(f"Using {n_tiles} existing tiles from {self.working_dir}")
        else:
            if use_existing:
                print("No existing tiles found; generating new tiles…")
            self._tile(pc)

        # 3. Dataset / DataLoader
        gravity_dim = self.cfg.datatransforms.kwargs.get("gravity_dim", 2)
        dataset = TileDataset(self.working_dir, gravity_dim=gravity_dim)
        loader = DataLoader(
            dataset,
            batch_size=self.params.get("batch_size", 4),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pyg,
        )

        # 4. Build and load model
        model = self._build_model()

        # 5. Inference loop
        feature_keys = self.cfg.get("feature_keys", "pos")
        output_list = []
        num_tiles = len(dataset)
        print(
            f"Running inference on {num_tiles} tiles (feature_keys='{feature_keys}')…"
        )

        with torch.no_grad():
            for batch in tqdm(loader, desc="Inferring"):
                batch = batch.to(self.device)

                # Build input features exactly as in test_pyg_variable.py:
                #   batch.x = get_features_by_keys(batch, feature_keys)
                # For feature_keys="pos"     → x = batch.pos         (N, 3)
                # For feature_keys="pos,x"   → x = cat(pos, feat)    (N, 3+C)
                # For feature_keys="pos,heights" → x = cat(pos, heights) (N, 4)
                batch.x = get_features_by_keys(batch, feature_keys)

                logits = model(batch)  # (N_batch, num_classes)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                pos = batch.pos.cpu().numpy()
                batch_ids = batch.batch.cpu().numpy()

                # PyG concatenates (3,) local_shift tensors → flat (B*3,) vector.
                # Reshape to (B, 3) for indexed access.
                shifts = batch.local_shift.cpu().numpy().reshape(-1, 3)

                for b in np.unique(batch_ids):
                    mask = batch_ids == b
                    # Restore global xyz:  xyz_orig = xyz_trans + local_shift
                    xyz = pos[mask] + shifts[b]  # (n, 3)
                    out = np.hstack([xyz, probs[mask]])  # (n, 3 + num_classes)
                    output_list.append(out)

        print(f"Inference done — {num_tiles} tiles processed")
        self.pred_cloud = np.vstack(output_list)  # (N_pred, 3+num_classes)

        # 6. Propagate labels to original full-resolution cloud
        knn_k = self.params.get("knn_k", 16)
        direct_threshold = self.params.get("direct_threshold", 0.05)
        prop_chunk_size = self.params.get("prop_chunk_size", 1_000_000)
        prop_n_jobs = self.params.get("prop_n_jobs", -1)
        self.output_labels, self.model_inferred = propagate_labels(
            self.pred_cloud,
            self.original_cloud,
            k=knn_k,
            direct_threshold=direct_threshold,
            chunk_size=prop_chunk_size,
            n_jobs=prop_n_jobs,
        )

        # 7. Save
        self._save()

        # 8. Optionally remove working tiles
        if self.params.get("delete_working_dir", True):
            shutil.rmtree(self.working_dir, ignore_errors=True)

        elapsed = time.time() - self.start_time
        print(f"Semantic segmentation took {elapsed:.1f} s")
        print("Semantic segmentation done")


# ─── Batch processing ─────────────────────────────────────────────────────────


def run_batch_inference(input_path: str, args):
    """Process all .las/.laz files in a directory or a single file.

    Args:
        input_path: Path to a single file or directory
        args: Parsed command line arguments
    """
    input_path_obj = Path(input_path)

    if input_path_obj.is_file():
        # Single file mode
        files_to_process = [input_path_obj]
    elif input_path_obj.is_dir():
        # Directory mode - find all .las and .laz files
        files_to_process = sorted(
            list(input_path_obj.glob("*.las")) + list(input_path_obj.glob("*.laz"))
        )
        if not files_to_process:
            print(f"No .las or .laz files found in {input_path}")
            return
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        return

    print(f"Processing {len(files_to_process)} file(s)…\n")

    start_time_batch = time.time()
    successful = 0
    failed = 0

    # Track metadata for CSV report
    results = []

    for i, file_path in enumerate(files_to_process, 1):
        print(f"[{i}/{len(files_to_process)}] {file_path.name}")

        file_start_time = time.time()
        status = "FAILED"
        output_path = None
        error_msg = None

        try:
            params = {
                "point_cloud_filename": str(file_path),
                "checkpoint_path": args.checkpoint,
                "cfg_path": args.cfg,
                "batch_size": args.batch_size,
                "use_cpu": args.cpu,
                "delete_working_dir": not args.keep_tiles,
                "label_offset": args.label_offset,
                "knn_k": args.knn_k,
                "direct_threshold": args.direct_threshold,
                "voxel_max": args.voxel_max,
                "prop_chunk_size": args.prop_chunk_size,
                "prop_n_jobs": args.prop_n_jobs,
                "use_existing_tiles": args.use_existing_tiles,
            }

            # Only override tile parameters if explicitly provided
            if args.tile_size is not None:
                params["tile_size"] = args.tile_size
            if args.tile_overlap is not None:
                params["tile_overlap"] = args.tile_overlap

            seg = SemanticSegmentation(params)
            seg.inference()
            status = "SUCCESS"
            output_path = seg.output_las
            successful += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            error_msg = str(e)
            failed += 1

        file_elapsed = time.time() - file_start_time
        results.append(
            {
                "filename": file_path.name,
                "input_path": str(file_path),
                "output_path": output_path if output_path else "N/A",
                "status": status,
                "time_seconds": file_elapsed,
                "error": error_msg if error_msg else "",
            }
        )

        print()  # blank line between files

    elapsed_batch = time.time() - start_time_batch

    # Save results to CSV
    df = pd.DataFrame(results)
    csv_output = (
        Path(input_path_obj if input_path_obj.is_dir() else input_path_obj.parent)
        / "inference_results.csv"
    )
    df.to_csv(csv_output, index=False)
    print(f"Results saved to: {csv_output}")

    print(f"\n{'=' * 70}")
    print(f"Batch processing complete: {successful}/{len(files_to_process)} successful")
    if failed > 0:
        print(f"Failed: {failed}")
    print(f"Total time: {elapsed_batch:.1f} s")
    print(f"Average time per file: {elapsed_batch / len(files_to_process):.1f} s")
    print(f"{'=' * 70}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PointNeXt semantic segmentation inference on LAS/LAZ files"
    )
    parser.add_argument(
        "--cfg",
        required=True,
        help="YAML config (e.g. cfgs/runpod-4090/forinstancev2/pointnext-s-pyg.yaml)",
    )
    parser.add_argument(
        "--input", required=True, help="Input LAS/LAZ file or directory of files"
    )
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pth)")
    parser.add_argument(
        "--output",
        default=None,
        help="Override output .las path (default: <input>.segmented.las)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Tile batches per forward pass (default: 4)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only inference")
    parser.add_argument(
        "--keep_tiles",
        action="store_true",
        help="Keep intermediate tile .npy files after inference",
    )
    parser.add_argument(
        "--use_existing_tiles",
        action="store_true",
        help="Reuse existing tiles if available; generate if missing",
    )
    parser.add_argument(
        "--voxel_max",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--tile_size",
        type=float,
        default=None,
        help="Tile edge in metres (overrides cfg)",
    )
    parser.add_argument(
        "--tile_overlap",
        type=float,
        default=None,
        help="Tile overlap ratio 0-1 (overrides cfg)",
    )
    parser.add_argument(
        "--label_offset",
        type=int,
        default=0,
        help="Add to predicted class IDs before saving "
        "(e.g. 1 to restore 1-indexed LAS classes, default: 0)",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=16,
        help="Neighbours for label propagation (default: 16)",
    )
    parser.add_argument(
        "--direct_threshold",
        type=float,
        default=0.05,
        help="Max distance (m) to nearest model point for a point to be marked "
        "model_inferred=1 in the output LAS (default: 0.05)",
    )
    parser.add_argument(
        "--prop_chunk_size",
        type=int,
        default=1_000_000,
        help="Points per chunk during label propagation kNN query (default: 1000000). "
        "Reduce to lower peak RAM usage.",
    )
    parser.add_argument(
        "--prop_n_jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for label propagation (-1 = all cores, default: -1). "
        "Set to 1 to disable parallelization.",
    )
    args = parser.parse_args()

    # Check if input is a directory (batch mode) or file (single mode)
    input_path = Path(args.input)
    if input_path.is_dir():
        # Batch mode: process all .las/.laz files in the directory
        run_batch_inference(args.input, args)
    else:
        # Single file mode: maintain backward compatibility
        params = {
            "point_cloud_filename": args.input,
            "checkpoint_path": args.checkpoint,
            "cfg_path": args.cfg,
            "batch_size": args.batch_size,
            "use_cpu": args.cpu,
            "delete_working_dir": not args.keep_tiles,
            "label_offset": args.label_offset,
            "knn_k": args.knn_k,
            "direct_threshold": args.direct_threshold,
            "voxel_max": args.voxel_max,
            "prop_chunk_size": args.prop_chunk_size,
            "prop_n_jobs": args.prop_n_jobs,
            "use_existing_tiles": args.use_existing_tiles,
        }
        if args.tile_size is not None:
            params["tile_size"] = args.tile_size
        if args.tile_overlap is not None:
            params["tile_overlap"] = args.tile_overlap

        seg = SemanticSegmentation(params)

        if args.output is not None:
            seg.output_las = args.output

        seg.inference()
