# PigBackfatFusion

**Sorting-system multimodal data fusion for backfat thickness prediction in finishing pigs**

PigBackfatFusion provides the implementation of a multimodal gated fusion model for non-contact backfat thickness prediction in finishing pigs. The model integrates segmented pig-back RGB images, age and body weight information, and pig-back point cloud features constructed from aligned RGB-D data.

This repository is associated with the manuscript:

> Sorting-system multimodal data fusion for backfat thickness prediction in finishing pigs

## Overview

Backfat thickness is an important indicator of fat deposition in finishing pigs, but conventional ultrasound measurement is labor-intensive and difficult to integrate into continuous sorting-system monitoring. This project explores a non-contact prediction approach using multimodal data collected as pigs pass through a sorting-based phenotypic acquisition system.

The complete method includes:

1. Synchronous acquisition of RGB images, depth images, and body weight in a sorting system.
2. Manual P2 backfat thickness measurement by ultrasound as the ground-truth label.
3. Pig-region segmentation using YOLO11n-seg.
4. Pig-back point cloud construction by applying RGB-derived masks to aligned depth images.
5. Backfat regression using RGB, growth-related meta-information, point cloud features, and gated multimodal fusion.

This repository contains the code for the **multimodal fusion regression model**, a small real experimental data example, normalization-parameter files, and the expected location for pretrained branch weights. The full open-source dataset is released separately in the same processed-data organization as `data_example/`, whereas the complete raw RGB-D videos are not stored in this repository because of their large size.

## Main features

- RGB and meta-information branch based on MobileNetV3-Small.
- Point cloud branch based on PointNet++ SSG.
- Gated feature fusion for adaptive integration of 2D appearance and 3D geometric features.
- Support for age and body weight ablation through command-line switches.
- Training-set-based normalization for RGB images, age, body weight, and backfat labels.
- Frame-level validation and output of prediction results.

## Method summary

The final fusion model takes three types of input:

- Segmented pig-back RGB image
- Pig-back point cloud with 2,048 points
- Age and body weight as two-dimensional meta-information

The RGB image and meta-information are encoded by the RGB-meta branch, while the point cloud is encoded by the PointNet++ branch. The two high-level features are projected into a unified 256-dimensional semantic space and fused through a gated feature fusion module. The fused feature is then passed to the regression head to predict backfat thickness in millimeters.

## Repository structure

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
│   ├── rgb_meta_best_model.pth          # optional/pretrained RGB-meta branch checkpoint
│   └── pointcloud_best_model.pth        # optional/pretrained point cloud branch checkpoint
└── data_example/
    ├── kg_data.txt                      # body-weight record for the example acquisition unit
    ├── rgb_pngs/                        # extracted RGB frames
    ├── depth_pngs/                      # extracted depth frames
    ├── mask/                            # pig-region segmentation masks
    ├── segmented_rgb/                   # segmented pig-back RGB images
    ├── segmented_depth/                 # segmented depth images
    ├── segmented_ply/                   # pig-back point clouds before fixed-size sampling
    └── 2048_pointcloud/                 # downsampled 2048-point pig-back point clouds
```

Note: the current fusion model only directly depends on `models/pointnet2_utils.py`. Other PointNet/PointNet++ files are retained as reference files from the original backbone implementation.

## Installation

Create a Python environment and install dependencies:

```bash
conda create -n pigbackfat python=3.10 -y
conda activate pigbackfat
pip install -r requirements.txt
```

For GPU training, install mutually compatible PyTorch and torchvision versions that match your CUDA and driver environment. If torchvision raises errors related to missing custom operators such as `torchvision::nms`, reinstall PyTorch and torchvision from the same release channel. The edge-device experiment in the manuscript was conducted on a Jetson Orin NX SUPER 16GB platform with Ubuntu 22.04.5 LTS, Jetson Linux 36.4.7, CUDA 12.6, cuDNN 9.16.0.29, and PyTorch 2.5.0a0+872d972e41.nv24.08.

## Data example

The `data_example/` directory contains a small real experimental sample selected from the sorting-system acquisition process. It is provided to demonstrate the processed-data organization, intermediate preprocessing outputs, body-weight record, and final model-input format used in this project.

The example follows the same organization as the released open-source dataset:

```text
data_example/
├── kg_data.txt                      # body-weight record for the example acquisition unit
├── rgb_pngs/                        # extracted RGB frames
├── depth_pngs/                      # extracted depth frames
├── mask/                            # pig-region segmentation masks generated from RGB images
├── segmented_rgb/                   # segmented pig-back RGB images
├── segmented_depth/                 # segmented depth images after mask filtering
├── segmented_ply/                   # pig-back point clouds before fixed-size sampling
└── 2048_pointcloud/                 # downsampled 2048-point pig-back point clouds used by the model
```

The files in `data_example/` are real examples from the experimental workflow, but this directory contains only a small subset and is not sufficient to reproduce the performance reported in the manuscript. Its purpose is to help users understand the expected data structure and the relationship among RGB frames, depth frames, segmentation masks, point clouds, and body-weight records.

The fusion model does not directly take raw RGB-D videos as input. For training and validation, each frame-level sample should be indexed by a CSV file that links the segmented RGB image, the corresponding 2048-point point cloud, the backfat label, age, and body weight. The body-weight value can be obtained from the matched `kg_data.txt` file or from a prepared metadata table for the corresponding acquisition unit.

## Input CSV format

The training and validation CSV files should contain one row per frame-level sample. The code supports either `image_path` or `rgb_path` for RGB images, and either `pcd_path` or `ply_path` for point cloud files.

Required columns:

| Column | Description |
|---|---|
| `image_path` or `rgb_path` | Path to the segmented pig-back RGB image |
| `pcd_path` or `ply_path` | Path to the 2048-point pig-back point cloud file in `.ply` format |
| `label` | Ground-truth P2 backfat thickness in mm |
| `age` | Pig age or growth-day information |
| `weight` | Body weight of the pig |

Optional columns:

| Column | Description |
|---|---|
| `pig_id` | Individual pig ID |
| `date` | Measurement or acquisition date |
| `folder_name` | Original folder or acquisition-unit name |
| `frame_id` | Frame index or frame name |

Example:

```csv
image_path,pcd_path,label,age,weight,pig_id,date,folder_name,frame_id
data_example/segmented_rgb/frame0000.png,data_example/2048_pointcloud/frame0000.ply,11.2,175,98.5,example_pig,2025-12-20,example_folder,frame0000
data_example/segmented_rgb/frame0001.png,data_example/2048_pointcloud/frame0001.ply,11.2,175,99.1,example_pig,2025-12-20,example_folder,frame0001
```

The `label` value should correspond to the manually measured P2 backfat thickness of the relevant pig-date unit. The `weight` value should correspond to the body-weight record matched to the same acquisition unit, which can be obtained from `kg_data.txt` or from a consolidated metadata file.

## Pretrained branch weights

The final fusion model in the manuscript was trained using a stage-wise strategy. The RGB-meta branch and the point cloud branch were first pretrained separately, and then their weights were loaded to initialize the fusion model.

```text
weight/
├── rgb_meta_best_model.pth
└── pointcloud_best_model.pth
```

These two files correspond to:

| File | Description |
|---|---|
| `rgb_meta_best_model.pth` | Best checkpoint of the RGB + age + body weight branch |
| `pointcloud_best_model.pth` | Best checkpoint of the PointNet++ point cloud branch |

The `weight/` directory contains pretrained branch checkpoints and normalization-statistics files. The `.pth` files are model weights, whereas `norm_stats.json` and `train_rgb_stats.json` store training-set normalization parameters used for input normalization and prediction denormalization.

## Training

### 1. Train the full multimodal fusion model

The training command used for the final model should enable RGB, point cloud, age, and body weight inputs:

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

This command assumes that the two pretrained branch checkpoints have been placed under `weight/`. If the checkpoints are not available, set `--rgb_ckpt ""`, `--point_ckpt ""`, `--freeze_rgb_branch 0`, and `--freeze_point_branch 0` for code testing or redevelopment.

### 2. Train without pretrained branch weights

For code testing or redevelopment, the fusion model can also be trained without pretrained branch checkpoints. In this case, do not freeze the two branches:

```bash
python train_regression_fusion_rgb_meta_pointnet2.py \
  --train_csv ./data/train_index.csv \
  --val_csv ./data/val_index.csv \
  --out_dir ./outputs/debug_run \
  --use_age 1 \
  --use_weight 1 \
  --fusion_type gate_feature \
  --rgb_ckpt "" \
  --point_ckpt "" \
  --freeze_rgb_branch 0 \
  --freeze_point_branch 0 \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 0 \
  --use_cpu 1 \
  --rgb_pretrained 0
```

This command is only intended to verify that the code and data paths are valid. It is not intended to reproduce the model performance reported in the manuscript.

## Evaluation

To evaluate a trained checkpoint on the validation set:

```bash
python train_regression_fusion_rgb_meta_pointnet2.py \
  --train_csv ./data/train_index.csv \
  --val_csv ./data/val_index.csv \
  --out_dir ./outputs/fusion_full \
  --resume ./outputs/fusion_full/best_model.pth \
  --eval_only 1 \
  --use_age 1 \
  --use_weight 1 \
  --fusion_type gate_feature
```

The script reports MAE, RMSE, MAPE, and R². Prediction results are saved as CSV files in the output directory.

## Ablation settings

The script supports switching age and body weight information on or off:

| Setting | Command-line options |
|---|---|
| RGB + point cloud | `--use_age 0 --use_weight 0` |
| RGB + point cloud + age | `--use_age 1 --use_weight 0` |
| RGB + point cloud + body weight | `--use_age 0 --use_weight 1` |
| RGB + point cloud + age + body weight | `--use_age 1 --use_weight 1` |

For the single RGB, RGB-meta, and point-cloud-only baselines reported in the manuscript, separate branch-level training scripts or equivalent model configurations are required.

## Dataset used in the manuscript

The dataset was collected from 34 finishing pigs during the final 40 days before market. The sorting-based acquisition system synchronously collected RGB videos, depth videos, and body weight as pigs passed through the system. P2 backfat thickness was measured every 3 days using ultrasound and used as the ground-truth label.

After screening and uniform frame sampling, the final regression dataset contained 10,570 paired RGB-depth samples. Pig-level splitting was used to avoid overlap of individual pigs between the training and validation sets.

## Open-source dataset

The open-source dataset released with this project is organized in the same processed-data format as `data_example/`. The released dataset contains processed samples for the complete set used in the study, including extracted RGB frames, extracted depth frames, pig-region masks, segmented RGB images, segmented depth images, point cloud files, 2048-point downsampled point clouds, and matched body-weight records.

Public dataset link:

- Dataset file: `data.zip`
- Baidu Netdisk: https://pan.baidu.com/s/1oI9Xt2JQVXEGdZxJNgKLqA
- Access code: `2avv`

The processed dataset is designed to support model training, validation, and redevelopment without requiring users to repeat the full raw-video decoding and preprocessing pipeline. The 2048-point point clouds in `2048_pointcloud/` are the direct point cloud inputs used by the point cloud branch of the fusion model. The files in `segmented_ply/` provide the corresponding pig-back point clouds before fixed-size sampling.

The complete raw RGB-D video data are not included in the open-source dataset because of their large storage size, approximately 940 GB. Raw RGB videos, raw depth videos, and additional acquisition information can be provided by the original authors upon reasonable request, subject to publication requirements, farm-data management rules, and reasonable-use conditions.

For access to the raw data or additional information, please contact the corresponding author:

```text
[Corresponding author name and email to be updated]
```

When requesting raw data, please include the intended research purpose, the requested data range, and affiliation information.

## Reported performance

### Backfat prediction performance of the final fusion model

| Evaluation level | MAE (mm) | RMSE (mm) | MAPE (%) | R² |
|---|---:|---:|---:|---:|
| Frame level | 0.7419 | 0.9413 | 6.3109 | 0.7799 |
| Pig-date level | 0.6704 | 0.8523 | 5.6798 | 0.8196 |
| Pig level | 0.5513 | 0.7107 | 4.6250 | 0.7114 |

### Input-combination comparison at the frame level

| RGB | Point cloud | Age | Body weight | MAE (mm) | RMSE (mm) | MAPE (%) | R² |
|---|---|---|---|---:|---:|---:|---:|
| ✓ | × | × | × | 0.9479 | 1.2074 | 8.0306 | 0.6378 |
| ✓ | × | ✓ | × | 0.9093 | 1.2263 | 7.6499 | 0.6264 |
| ✓ | × | × | ✓ | 0.9228 | 1.1806 | 7.9866 | 0.6547 |
| ✓ | × | ✓ | ✓ | 0.8965 | 1.1327 | 7.7274 | 0.6813 |
| × | ✓ | × | × | 0.8063 | 0.9887 | 6.9012 | 0.7572 |
| ✓ | ✓ | × | × | 0.7683 | 0.9591 | 6.5319 | 0.7715 |
| ✓ | ✓ | ✓ | × | 0.7551 | 0.9445 | 6.4379 | 0.7793 |
| ✓ | ✓ | × | ✓ | 0.7634 | 0.9626 | 6.5756 | 0.7698 |
| ✓ | ✓ | ✓ | ✓ | 0.7419 | 0.9413 | 6.3109 | 0.7799 |

### Edge deployment result

On a Jetson Orin NX SUPER 16GB, the full pipeline achieved an average end-to-end inference time of 1200.3 ms per sample. Point cloud downsampling and point cloud feature extraction were the main runtime bottlenecks.

## License

This project is released under the MIT License. See `LICENSE` for details.

The PointNet/PointNet++ backbone-related files are adapted from public PointNet++ implementations. Please also respect the license terms of the original implementation.

## Citation

If you use this repository, please cite the associated paper after publication:

```bibtex
@article{zhang2026pigbackfatfusion,
  title   = {Sorting-system multimodal data fusion for backfat thickness prediction in finishing pigs},
  author  = {Zhang, Zian and Li, Peng and Wu, Hongwei and Wang, Kelin and Liu, Longshen and Shen, Mingxia},
  journal = {To be updated},
  year    = {2026}
}
```

## Acknowledgements

This work was supported by the National Key R&D Program of China.
