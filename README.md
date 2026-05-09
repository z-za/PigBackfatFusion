# PigBackfatFusion

**Backfat thickness prediction in finishing pigs using multimodal information fusion in a sorting system**

PigBackfatFusion provides the implementation of a multimodal gated fusion model for non-contact backfat thickness prediction in finishing pigs. The model uses segmented pig-back RGB images, age and body weight information, and 2048-point pig-back point clouds constructed from aligned RGB-D data.

This repository is associated with the manuscript:

> Backfat thickness prediction in finishing pigs using multimodal information fusion in a sorting system

## Repository contents

```text
PigBackfatFusion/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── fusion_dataset_rgb_meta_pointnet2.py
├── model_regression_fusion_rgb_meta_pointnet2.py
├── train_regression_fusion_rgb_meta_pointnet2.py
├── models/
│   ├── pointnet2_utils.py
│   └── other PointNet/PointNet++ reference modules
├── weight/
│   ├── norm_stats.json
│   ├── train_rgb_stats.json
│   ├── rgb_meta_best_model.pth
│   └── pointcloud_best_model.pth
└── data_example/
    ├── kg_data.txt
    ├── rgb_pngs/
    ├── depth_pngs/
    ├── mask/
    ├── segmented_rgb/
    ├── segmented_depth/
    ├── segmented_ply/
    └── 2048/
```

The repository contains:

- model code for multimodal gated fusion regression;
- PointNet++ utility modules used by the point cloud branch;
- pretrained branch weights;
- normalization statistics used during training and inference;
- a small real experimental example in `data_example/`.

The current fusion model directly depends on `models/pointnet2_utils.py`. Other PointNet/PointNet++ files are retained as reference modules.

## Installation

```bash
conda create -n pigbackfat python=3.10 -y
conda activate pigbackfat
pip install -r requirements.txt
```

For GPU training, install PyTorch and torchvision versions that match your CUDA and driver environment.

## Data example

The `data_example/` directory contains one small real experimental example selected from the sorting-system acquisition process. It is provided only to demonstrate the processed-data organization and input format.

```text
data_example/
├── kg_data.txt          # body-weight record for this example acquisition unit
├── rgb_pngs/            # extracted RGB frames
├── depth_pngs/          # extracted depth frames
├── mask/                # pig-region segmentation masks
├── segmented_rgb/       # segmented pig-back RGB images
├── segmented_depth/     # segmented depth images
├── segmented_ply/       # pig-back point clouds before fixed-size sampling
└── 2048/                # downsampled 2048-point pig-back point clouds
```

The `data_example/` folder is not sufficient to reproduce the results reported in the manuscript.

## Open-source dataset

The processed dataset is released separately:

- Dataset file: `data.zip`
- Baidu Netdisk: `https://pan.baidu.com/s/13mFfYMCwcRSnhUNJx8BzTw`
- Access code: `trhf`

The current public dataset contains processed visual and geometric samples, including:

```text
<sample_folder>/
├── rgb_pngs/
├── depth_pngs/
├── mask/
├── segmented_rgb/
├── segmented_depth/
├── segmented_ply/
└── 2048/
```

The public dataset does not currently include the ground-truth P2 backfat values, body-weight values, age information, or the CSV index files used for supervised training and validation. These metadata and CSV files will be released until the paper is published.

The complete raw RGB-D video data, totaling approximately 940 GB, are not included in the public dataset because of storage-size limitations. Raw RGB videos, raw depth videos, and additional acquisition information can be provided by the original authors upon reasonable request, subject to publication requirements, farm-data management rules, and reasonable-use conditions.

For access to the raw data or additional information, please contact the corresponding author listed in the associated manuscript.

## Input CSV format

For training and validation, each frame-level sample should be indexed by a CSV file linking the segmented RGB image, the 2048-point point cloud, the ground-truth backfat label, age, and body weight.

Required columns:

| Column | Description |
|---|---|
| `image_path` or `rgb_path` | Path to the segmented pig-back RGB image |
| `pcd_path` or `ply_path` | Path to the 2048-point pig-back point cloud file |
| `label` | Ground-truth P2 backfat thickness in mm |
| `age` | Pig age or growth-day information |
| `weight` | Body weight of the pig |

Optional columns include `pig_id`, `date`, `folder_name`, and `frame_id`.

The true labels, age values, body-weight values, and the CSV files used in the manuscript will be released until the paper is published.

## Pretrained weights and normalization files

The `weight/` directory contains pretrained branch checkpoints and normalization-statistics files:

```text
weight/
├── norm_stats.json
├── train_rgb_stats.json
├── rgb_meta_best_model.pth
└── pointcloud_best_model.pth
```

The `.pth` files are model weights. `norm_stats.json` and `train_rgb_stats.json` store training-set normalization parameters used for input normalization and prediction denormalization.

## Training

Train the full multimodal fusion model:

```bash
python train_regression_fusion_rgb_meta_pointnet2.py \
  --train_csv ./data/train_index.csv \
  --val_csv ./data/val_index.csv \
  --out_dir ./outputs/fusion_full \
  --image_h 480 \
  --image_w 640 \
  --num_points 2048 \
  --use_age 1 \
  --use_weight 1 \
  --fusion_type gate_feature \
  --rgb_ckpt ./weight/rgb_meta_best_model.pth \
  --point_ckpt ./weight/pointcloud_best_model.pth \
  --freeze_rgb_branch 1 \
  --freeze_point_branch 1 \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-3 \
  --weight_decay 1e-4
```

This command requires `train_index.csv` and `val_index.csv`. These files will be released until the paper is published.

## License

This project is released under the MIT License. See `LICENSE` for details.

The PointNet/PointNet++ backbone-related files are adapted from public PointNet++ implementations. Please also respect the license terms of the original implementation.

## Citation

If you use this repository, please cite the associated paper after publication:

```bibtex
@article{zhang2026pigbackfatfusion,
  title   = {Backfat thickness prediction in finishing pigs using multimodal information fusion in a sorting system},
  author  = {Zhang, Zian and Li, Peng and Wu, Hongwei and Wang, Kelin and Liu, Longshen and Shen, Mingxia},
  journal = {To be updated},
  year    = {2026}
}
```