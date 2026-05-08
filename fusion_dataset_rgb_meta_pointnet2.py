# -*- coding: utf-8 -*-
"""
fusion_dataset_rgb_meta_pointnet2.py

融合数据集：RGB + (age, weight) + PointNet++ 输入点云

特点：
1. 支持 image_path/rgb_path 与 pcd_path/ply_path 双列名兼容；
2. label / age / weight / rgb mean-std 全部使用训练集统计量；
3. age / weight 始终返回，模型内部用开关决定是否启用，便于做消融时直接复用同一份 csv 与同一份 rgb+age+weight ckpt；
4. 点云默认只做 x、y 中心化，z 保持原始值，与现有 PointNet++ 单分支保持一致。
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import open3d as o3d


# =========================
# 基础工具
# =========================
def normalize_pig_id(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        return str(int(s))
    return s



def normalize_date(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s



def _resolve_image_col(df: pd.DataFrame) -> str:
    if "image_path" in df.columns:
        return "image_path"
    if "rgb_path" in df.columns:
        return "rgb_path"
    raise ValueError("CSV 中缺少 image_path 或 rgb_path 列。")



def _resolve_point_col(df: pd.DataFrame) -> str:
    if "pcd_path" in df.columns:
        return "pcd_path"
    if "ply_path" in df.columns:
        return "ply_path"
    raise ValueError("CSV 中缺少 pcd_path 或 ply_path 列。")



def compute_scalar_stats(csv_path: str, col_name: str) -> Tuple[float, float]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if col_name not in df.columns:
        raise ValueError(f"{csv_path} 中缺少列: {col_name}")
    vals = pd.to_numeric(df[col_name], errors="coerce").dropna().astype(np.float32).values
    if len(vals) == 0:
        raise ValueError(f"{csv_path} 中列 {col_name} 没有有效数值。")
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    if std < 1e-12:
        std = 1.0
    return mean, std



def compute_rgb_stats(
    csv_path: str,
    image_size: Tuple[int, int] = (480, 640),
    max_samples: int = 0,
    cache_json: Optional[str] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    仅用训练集图像统计 RGB 均值与标准差。
    统计时会先 resize 到网络输入尺寸，再转为 [0,1] 浮点。

    参数:
        max_samples = 0 表示使用全部训练图像；>0 时随机均匀截取前 max_samples 张（按 CSV 顺序）。
    """
    if cache_json is not None and os.path.isfile(cache_json):
        with open(cache_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        mean = tuple(float(x) for x in obj["mean"])
        std = tuple(float(x) for x in obj["std"])
        return mean, std

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    image_col = _resolve_image_col(df)
    image_paths = [str(p).strip() for p in df[image_col].tolist() if str(p).strip()]
    if len(image_paths) == 0:
        raise ValueError(f"{csv_path} 中没有有效图像路径。")

    if max_samples > 0:
        image_paths = image_paths[:max_samples]

    h, w = int(image_size[0]), int(image_size[1])
    resize = transforms.Resize((h, w))

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for p in image_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"统计 RGB 均值方差时未找到图像: {p}")
        img = Image.open(p).convert("RGB")
        img = resize(img)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
        arr = arr.reshape(-1, 3)
        channel_sum += arr.sum(axis=0)
        channel_sum_sq += (arr ** 2).sum(axis=0)
        total_pixels += arr.shape[0]

    mean = channel_sum / max(total_pixels, 1)
    var = channel_sum_sq / max(total_pixels, 1) - mean ** 2
    var = np.clip(var, 1e-12, None)
    std = np.sqrt(var)

    mean_t = tuple(float(x) for x in mean.tolist())
    std_t = tuple(float(x) for x in std.tolist())

    if cache_json is not None:
        os.makedirs(os.path.dirname(cache_json), exist_ok=True)
        with open(cache_json, "w", encoding="utf-8") as f:
            json.dump({"mean": list(mean_t), "std": list(std_t)}, f, ensure_ascii=False, indent=2)

    return mean_t, std_t



def build_rgb_transform(image_size=(480, 640), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# =========================
# Dataset
# =========================
class FusionRGBMetaPointNet2Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_size: Tuple[int, int] = (480, 640),
        rgb_mean: Sequence[float] = (0.485, 0.456, 0.406),
        rgb_std: Sequence[float] = (0.229, 0.224, 0.225),
        num_points: int = 2048,
        strict_num_points: bool = True,
        normalize_label: bool = False,
        label_mean: Optional[float] = None,
        label_std: Optional[float] = None,
        normalize_age: bool = False,
        age_mean: Optional[float] = None,
        age_std: Optional[float] = None,
        normalize_weight: bool = False,
        weight_mean: Optional[float] = None,
        weight_std: Optional[float] = None,
        check_files: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"未找到 CSV: {csv_path}")

        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig").copy()
        self.image_col = _resolve_image_col(self.df)
        self.point_col = _resolve_point_col(self.df)
        self.num_points = int(num_points)
        self.strict_num_points = bool(strict_num_points)
        self.normalize_label = bool(normalize_label)
        self.normalize_age = bool(normalize_age)
        self.normalize_weight = bool(normalize_weight)
        self.label_mean = 0.0 if label_mean is None else float(label_mean)
        self.label_std = 1.0 if label_std is None else float(label_std)
        self.age_mean = 0.0 if age_mean is None else float(age_mean)
        self.age_std = 1.0 if age_std is None else float(age_std)
        self.weight_mean = 0.0 if weight_mean is None else float(weight_mean)
        self.weight_std = 1.0 if weight_std is None else float(weight_std)
        self.dtype = dtype

        if "label" not in self.df.columns:
            raise ValueError(f"{csv_path} 中缺少 label 列")
        if "age" not in self.df.columns:
            self.df["age"] = 0.0
        if "weight" not in self.df.columns:
            self.df["weight"] = 0.0

        for c in [self.image_col, self.point_col]:
            self.df[c] = self.df[c].astype(str).str.strip()
        self.df["label"] = pd.to_numeric(self.df["label"], errors="coerce")
        self.df["age"] = pd.to_numeric(self.df["age"], errors="coerce")
        self.df["weight"] = pd.to_numeric(self.df["weight"], errors="coerce")

        for c in ["pig_id", "date", "folder_name"]:
            if c not in self.df.columns:
                self.df[c] = ""
        self.df["pig_id"] = self.df["pig_id"].apply(normalize_pig_id)
        self.df["date"] = self.df["date"].apply(normalize_date)
        self.df["folder_name"] = self.df["folder_name"].astype(str).str.strip()

        self.df = self.df.dropna(subset=[self.image_col, self.point_col, "label", "age", "weight"]).reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"{csv_path} 清洗后没有可用样本。")

        if abs(self.label_std) < 1e-12:
            self.label_std = 1.0
        if abs(self.age_std) < 1e-12:
            self.age_std = 1.0
        if abs(self.weight_std) < 1e-12:
            self.weight_std = 1.0

        self.transform = build_rgb_transform(image_size=image_size, mean=rgb_mean, std=rgb_std)

        if check_files:
            missing_img, missing_pcd = [], []
            for _, row in self.df.iterrows():
                ip = str(row[self.image_col])
                pp = str(row[self.point_col])
                if not os.path.isfile(ip):
                    missing_img.append(ip)
                if not os.path.isfile(pp):
                    missing_pcd.append(pp)
            if missing_img:
                raise FileNotFoundError("图像文件缺失，前 10 个如下：\n" + "\n".join(missing_img[:10]))
            if missing_pcd:
                raise FileNotFoundError("点云文件缺失，前 10 个如下：\n" + "\n".join(missing_pcd[:10]))

    def __len__(self) -> int:
        return len(self.df)

    def _read_ply_xyz(self, ply_path: str) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"点云维度异常: {ply_path}, shape={points.shape}")
        if len(points) == 0:
            raise ValueError(f"空点云: {ply_path}")
        return points

    def _fix_num_points_if_needed(self, points: np.ndarray, ply_path: str) -> np.ndarray:
        n = points.shape[0]
        if n == self.num_points:
            return points
        if self.strict_num_points:
            raise ValueError(f"点数不等于期望值 {self.num_points}: {ply_path}, 实际点数={n}")
        if n > self.num_points:
            return points[:self.num_points]
        extra_idx = np.random.choice(n, size=self.num_points - n, replace=True)
        return np.concatenate([points, points[extra_idx]], axis=0)

    @staticmethod
    def _center_xy_only(points: np.ndarray) -> np.ndarray:
        points = points.copy()
        xy_mean = points[:, :2].mean(axis=0, keepdims=True)
        points[:, :2] = points[:, :2] - xy_mean
        return points

    def _norm_label(self, x: float) -> float:
        return (x - self.label_mean) / self.label_std if self.normalize_label else x

    def _norm_age(self, x: float) -> float:
        return (x - self.age_mean) / self.age_std if self.normalize_age else x

    def _norm_weight(self, x: float) -> float:
        return (x - self.weight_mean) / self.weight_std if self.normalize_weight else x

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_path = str(row[self.image_col])
        pcd_path = str(row[self.point_col])

        label_raw = float(row["label"])
        age_raw = float(row["age"])
        weight_raw = float(row["weight"])

        label = self._norm_label(label_raw)
        age = self._norm_age(age_raw)
        weight = self._norm_weight(weight_raw)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        points = self._read_ply_xyz(pcd_path)
        points = self._fix_num_points_if_needed(points, pcd_path)
        points = self._center_xy_only(points)

        meta = torch.tensor([age, weight], dtype=torch.float32)

        return {
            "image": image,
            "points": torch.as_tensor(points, dtype=self.dtype),  # (N,3)
            "meta": meta,  # [age, weight]，均为标准化后数值（若开启）
            "age": torch.tensor([age], dtype=torch.float32),
            "weight": torch.tensor([weight], dtype=torch.float32),
            "age_raw": torch.tensor([age_raw], dtype=torch.float32),
            "weight_raw": torch.tensor([weight_raw], dtype=torch.float32),
            "label": torch.tensor([label], dtype=torch.float32),
            "label_raw": torch.tensor([label_raw], dtype=torch.float32),
            "image_path": image_path,
            "rgb_path": image_path,
            "pcd_path": pcd_path,
            "ply_path": pcd_path,
            "pig_id": "" if row["pig_id"] is None else str(row["pig_id"]),
            "date": "" if row["date"] is None else str(row["date"]),
            "folder_name": "" if row["folder_name"] is None else str(row["folder_name"]),
        }



def create_dataset(
    csv_path: str,
    image_size: Tuple[int, int] = (480, 640),
    rgb_mean: Sequence[float] = (0.485, 0.456, 0.406),
    rgb_std: Sequence[float] = (0.229, 0.224, 0.225),
    num_points: int = 2048,
    strict_num_points: bool = True,
    normalize_label: bool = False,
    label_mean: Optional[float] = None,
    label_std: Optional[float] = None,
    normalize_age: bool = False,
    age_mean: Optional[float] = None,
    age_std: Optional[float] = None,
    normalize_weight: bool = False,
    weight_mean: Optional[float] = None,
    weight_std: Optional[float] = None,
    check_files: bool = True,
):
    return FusionRGBMetaPointNet2Dataset(
        csv_path=csv_path,
        image_size=image_size,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
        num_points=num_points,
        strict_num_points=strict_num_points,
        normalize_label=normalize_label,
        label_mean=label_mean,
        label_std=label_std,
        normalize_age=normalize_age,
        age_mean=age_mean,
        age_std=age_std,
        normalize_weight=normalize_weight,
        weight_mean=weight_mean,
        weight_std=weight_std,
        check_files=check_files,
    )


if __name__ == "__main__":
    pass
