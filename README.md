# PigBackfatFusion

**Backfat thickness prediction in finishing pigs using multimodal information fusion in a sorting system**

PigBackfatFusion provides the code for a multimodal regression model for non-contact backfat thickness prediction in finishing pigs. The model integrates segmented pig-back RGB images, age and body-weight information, and 2048-point pig-back point clouds constructed from aligned RGB-D data.

This repository is associated with the manuscript:

> Backfat thickness prediction in finishing pigs using multimodal information fusion in a sorting system

## Repository structure

```text
PigBackfatFusion/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── dataset.py
├── fusion_model.py
├── train.py
├── models/
│   └── pointnet2_utils.py
├── preprocess/
│   ├── generate_pointcloud.py
│   └── downsample_to_2048.py
├── configs/
│   ├── norm_stats.json
│   └── train_rgb_stats.json
├── weight/
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

The main files are:

- `dataset.py`: dataset loader for segmented RGB images, age/body-weight metadata, and 2048-point pig-back point clouds.
- `fusion_model.py`: MobileNetV3-based RGB-meta branch, PointNet++ point cloud branch, and multimodal fusion regression model.
- `train.py`: training and validation script for the fusion model.
- `models/pointnet2_utils.py`: PointNet++ set-abstraction layers used by the point cloud branch.
- `preprocess/generate_pointcloud.py`: generates segmented depth images and pig-back point clouds from RGB-D images and segmentation masks.
- `preprocess/downsample_to_2048.py`: converts segmented point clouds to fixed-size 2048-point point clouds.
- `configs/`: normalization statistics computed from the training set.
- `weight/`: pretrained single-branch checkpoints used to initialize the RGB-meta and point cloud branches.

## Installation

```bash
conda create -n pigbackfat python=3.10 -y
conda activate pigbackfat
pip install -r requirements.txt
```

For GPU training, install PyTorch and torchvision versions that match your CUDA and driver environment.

The preprocessing scripts also require OpenCV and OpenPyXL. If they are not included in your environment, install them manually:

```bash
pip install opencv-python openpyxl
```

## Data organization

The `data_example/` directory contains a small real example from the sorting-system acquisition process. It is provided to demonstrate the processed-data organization and input format.

```text
data_example/
├── kg_data.txt          # body-weight records for this example acquisition unit
├── rgb_pngs/            # extracted RGB frames
├── depth_pngs/          # extracted depth frames
├── mask/                # pig-region segmentation masks
├── segmented_rgb/       # segmented pig-back RGB images
├── segmented_depth/     # segmented depth images
├── segmented_ply/       # pig-back point clouds before fixed-size sampling
└── 2048/                # 2048-point pig-back point clouds
```

## Preprocessing

### 1. Generate segmented depth images and point clouds

`preprocess/generate_pointcloud.py` converts RGB-D images and segmentation masks into segmented depth images and colored point clouds.

### 2. Downsample point clouds to 2048 points

`preprocess/downsample_to_2048.py` processes the point clouds in `yolo_out/segmented_ply/` and saves fixed-size point clouds to `yolo_out/2048/`.

## Training

Train the multimodal fusion model:

```bash
python train.py \
  --train_csv ./data/train_index.csv \
  --val_csv ./data/val_index.csv \
  --out_dir ./outputs/fusion_full \
  --image_h 480 \
  --image_w 640 \
  --num_points 2048 \
  --strict_num_points 1 \
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

## Pretrained weights and normalization statistics

The `weight/` directory stores pretrained branch checkpoints:

```text
weight/
├── rgb_meta_best_model.pth
└── pointcloud_best_model.pth
```

The `configs/` directory stores training-set normalization statistics:

```text
configs/
├── norm_stats.json
└── train_rgb_stats.json
```

`norm_stats.json` contains the mean and standard deviation of the label, age, body weight, and RGB channels. `train_rgb_stats.json` contains only the RGB channel statistics.

## Public dataset

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

The public dataset does not currently include ground-truth P2 backfat values, body-weight values, age information, or the CSV index files used for supervised training and validation. These metadata and CSV files will be released after the paper is published.

The complete raw RGB-D video data are not included in the public dataset because of storage-size limitations. Raw RGB videos, raw depth videos, and additional acquisition information can be provided by the original authors upon reasonable request, subject to publication requirements, farm-data management rules, and reasonable-use conditions.

## Notes

- `data_example/` is only a format example and cannot reproduce the manuscript results.
- The fusion training script requires both segmented RGB images and 2048-point point clouds.
- The training and validation split used in the manuscript follows pig-level independence to avoid sample leakage between training and validation sets.

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
