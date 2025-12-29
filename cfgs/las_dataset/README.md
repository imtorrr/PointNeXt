# LASDataset Configuration

Universal dataset for LAS/LAZ point cloud files with automatic tiling.

## Directory Structure

Your data should be organized as follows:

```
data/LAS/
├── raw/
│   ├── train/
│   │   ├── file1.las
│   │   ├── file2.laz
│   │   └── ...
│   ├── val/
│   │   ├── file3.las
│   │   └── ...
│   └── test/
│       ├── file4.las
│       └── ...
├── tiled/          # Auto-generated tiles
└── processed/      # Auto-generated cached data
```

## Quick Start

### 1. Prepare Your Data

Place your LAS/LAZ files in the appropriate split folders:
- `data/LAS/raw/train/` - Training files
- `data/LAS/raw/val/` - Validation files
- `data/LAS/raw/test/` - Test files

### 2. Update Configuration

Edit `cfgs/las_dataset/default.yaml`:

```yaml
dataset:
  common:
    data_root: data/LAS  # Path to your data
    label_field: classification  # LAS field containing labels

num_classes: 4  # Update based on your dataset
```

### 3. Train

**For PyTorch Geometric models:**
```bash
python examples/segmentation/main_pyg.py \
  --cfg cfgs/las_dataset/default.yaml
```

**For standard models:**
```bash
python examples/segmentation/main.py \
  --cfg cfgs/las_dataset/default.yaml
```

## Available Configurations

### default.yaml
Standard configuration for training with labeled LAS data (XYZ only).

### unlabeled.yaml
Configuration for inference on new LAS files without labels.

## Configuration Options

### Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_root` | str | `data/LAS` | Root folder containing train/val/test |
| `split` | str | `train` | Which split to use |
| `tile_size` | float | `6.0` | Tile size in meters |
| `tile_overlap` | float | `0.5` | Overlap ratio (0-1) |
| `voxel_size` | float | `0.04` | Voxel downsampling size |
| `voxel_max` | int | `80000` | Max points per sample (applied during loading) |
| `min_points_per_tile` | int | `2000` | Min points to keep tile (applied during tiling) |
| `label_field` | str\|null | `classification` | LAS field for labels (null for unlabeled) |
| `label_offset` | int | `0` | Offset to remap labels (e.g., -1 maps [1,2,3] to [0,1,2]) |
| `use_rgb` | bool | `false` | Include RGB features |
| `use_intensity` | bool | `false` | Include intensity |
| `use_return_number` | bool | `false` | Include return info |

### Label Field Options

Common LAS label fields:
- `classification` - Standard LAS classification field (default)
- `label` - Custom label field
- `user_data` - User-defined field
- `null` - For unlabeled data (inference only)

### Label Remapping

If your labels need to be remapped (e.g., LAS files have labels [1,2,3] but PyTorch expects [0,1,2]), use the `label_offset` parameter:

```yaml
dataset:
  common:
    label_field: classification
    label_offset: -1  # Subtract 1 from all labels
```

This applies the offset during data loading, so you don't need to retile or modify your raw LAS files.

## Usage Examples

### Training (PyG)

```bash
python examples/segmentation/main_pyg.py \
  --cfg cfgs/las_dataset/default.yaml
```

### Inference on Unlabeled Data

```bash
python examples/segmentation/test.py \
  --cfg cfgs/las_dataset/unlabeled.yaml \
  --pretrained_path path/to/checkpoint.pth
```

### Custom Data Path

```bash
python examples/segmentation/main.py \
  --cfg cfgs/las_dataset/default.yaml \
  dataset.common.data_root=path/to/your/data
```

## First Run Behavior

On the first run, LASDataset will automatically:
1. Discover all LAS/LAZ files in the split folder
2. Tile each file into smaller chunks (saved to `tiled/` folder)
3. Optionally presample and cache (saved to `processed/` folder)

Subsequent runs will use the cached tiles for faster loading.

## Notes

- **XYZ coordinates are always included**
- Additional features (RGB, intensity, etc.) are optional and disabled by default
- Empty tiles are automatically filtered out
- Supports both `.las` and `.laz` (compressed) formats
- Tiling happens once and is cached for future use
- For large datasets, set `presample: true` to cache preprocessed data

## Troubleshooting

**Q: "No LAS/LAZ files found"**
A: Check that your files are in `data_root/raw/train/` (or val/test)

**Q: "Label field not found"**
A: Check your LAS file has the specified label field, or set `label_field: null`

**Q: Tiling is slow**
A: This only happens once. Subsequent runs will use cached tiles.

**Q: Out of memory**
A: Reduce `voxel_max` or `batch_size` in the config
