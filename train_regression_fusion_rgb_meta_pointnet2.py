# -*- coding: utf-8 -*-
"""
train_regression_fusion_rgb_meta_pointnet2.py

训练脚本：
1. PointNet++ + (RGB+Age+Weight) 融合；
2. age / weight 输入开关；
3. 可选两种融合头：
   - gate_feature
   - pred_weighted_residual
4. 训练集统计量用于：
   - label 标准化
   - rgb 标准化
   - age 标准化
   - weight 标准化
"""

from __future__ import annotations

import os
import csv
import json
import time
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from fusion_dataset_rgb_meta_pointnet2 import (
    create_dataset,
    compute_scalar_stats,
    compute_rgb_stats,
)
from model_regression_fusion_rgb_meta_pointnet2 import (
    build_fusion_regression_model,
    FusionRegressionLoss,
)


# =========================
# 基础工具
# =========================
def str2boolint(x) -> bool:
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    return x in {"1", "true", "yes", "y", "t"}



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)



def make_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)



def format_seconds(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"



def denormalize_labels(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return x * std + mean



def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}



def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def save_checkpoint(
    save_path: str,
    epoch: int,
    model,
    optimizer,
    scheduler,
    best_val_mae: float,
    norm_stats: Dict,
    args_dict: Dict,
):
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_mae": float(best_val_mae),
        "norm_stats": norm_stats,
        "args": args_dict,
    }, save_path)



def load_checkpoint(ckpt_path: str, model=None, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and ckpt.get("optimizer_state_dict", None) is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict", None) is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


# =========================
# 训练 / 验证
# =========================
def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out



def train_one_epoch(model, loader, optimizer, criterion, device, label_mean: float, label_std: float):
    model.train()
    total_loss = 0.0
    total_loss_main = 0.0
    total_loss_rgb = 0.0
    total_loss_pc = 0.0
    total_loss_delta_reg = 0.0
    total_samples = 0
    y_true_norm_all, y_pred_norm_all = [], []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        target = batch["label"]

        optimizer.zero_grad(set_to_none=True)
        pred, aux = model.forward_batch(batch)
        loss, detail = criterion(pred, target, aux)
        loss.backward()
        optimizer.step()

        bs = target.shape[0]
        total_samples += bs
        total_loss += float(loss.item()) * bs
        total_loss_main += float(detail["loss_main"].item()) * bs
        total_loss_rgb += float(detail["loss_rgb"].item()) * bs
        total_loss_pc += float(detail["loss_pc"].item()) * bs
        total_loss_delta_reg += float(detail["loss_delta_reg"].item()) * bs

        y_true_norm_all.append(target.detach().cpu().numpy().reshape(-1))
        y_pred_norm_all.append(pred.detach().cpu().numpy().reshape(-1))

    y_true_norm_all = np.concatenate(y_true_norm_all, axis=0).reshape(-1)
    y_pred_norm_all = np.concatenate(y_pred_norm_all, axis=0).reshape(-1)
    y_true = denormalize_labels(y_true_norm_all, label_mean, label_std)
    y_pred = denormalize_labels(y_pred_norm_all, label_mean, label_std)
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        "loss": total_loss / max(total_samples, 1),
        "loss_main": total_loss_main / max(total_samples, 1),
        "loss_rgb": total_loss_rgb / max(total_samples, 1),
        "loss_pc": total_loss_pc / max(total_samples, 1),
        "loss_delta_reg": total_loss_delta_reg / max(total_samples, 1),
    })
    return metrics


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device, label_mean: float, label_std: float):
    model.eval()
    total_loss = 0.0
    total_loss_main = 0.0
    total_loss_rgb = 0.0
    total_loss_pc = 0.0
    total_loss_delta_reg = 0.0
    total_samples = 0
    y_true_norm_all, y_pred_norm_all = [], []
    meta_rows = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        target = batch["label"]
        pred, aux = model.forward_batch(batch)
        loss, detail = criterion(pred, target, aux)

        bs = target.shape[0]
        total_samples += bs
        total_loss += float(loss.item()) * bs
        total_loss_main += float(detail["loss_main"].item()) * bs
        total_loss_rgb += float(detail["loss_rgb"].item()) * bs
        total_loss_pc += float(detail["loss_pc"].item()) * bs
        total_loss_delta_reg += float(detail["loss_delta_reg"].item()) * bs

        pred_np = pred.detach().cpu().numpy().reshape(-1)
        target_np = target.detach().cpu().numpy().reshape(-1)
        y_true_norm_all.append(target_np)
        y_pred_norm_all.append(pred_np)

        rgb_pred_np = aux["rgb_pred"].detach().cpu().numpy().reshape(-1) if aux.get("rgb_pred", None) is not None else None
        pc_pred_np = aux["pc_pred"].detach().cpu().numpy().reshape(-1) if aux.get("pc_pred", None) is not None else None
        base_pred_np = aux["base_pred"].detach().cpu().numpy().reshape(-1) if aux.get("base_pred", None) is not None else None
        delta_np = aux["delta"].detach().cpu().numpy().reshape(-1) if aux.get("delta", None) is not None else None
        conf_np = aux["conf"].detach().cpu().numpy() if aux.get("conf", None) is not None else None

        label_raw_np = batch["label_raw"].detach().cpu().numpy().reshape(-1)
        age_raw_np = batch["age_raw"].detach().cpu().numpy().reshape(-1)
        weight_raw_np = batch["weight_raw"].detach().cpu().numpy().reshape(-1)

        pred_raw_np = denormalize_labels(pred_np, label_mean, label_std)
        rgb_pred_raw_np = denormalize_labels(rgb_pred_np, label_mean, label_std) if rgb_pred_np is not None else None
        pc_pred_raw_np = denormalize_labels(pc_pred_np, label_mean, label_std) if pc_pred_np is not None else None
        base_pred_raw_np = denormalize_labels(base_pred_np, label_mean, label_std) if base_pred_np is not None else None
        delta_raw_np = delta_np * label_std if delta_np is not None else None

        for i in range(bs):
            row = {
                "image_path": batch["image_path"][i],
                "pcd_path": batch["pcd_path"][i],
                "pig_id": batch["pig_id"][i],
                "date": batch["date"][i],
                "folder_name": batch["folder_name"][i],
                "label": float(label_raw_np[i]),
                "pred": float(pred_raw_np[i]),
                "abs_err": float(abs(pred_raw_np[i] - label_raw_np[i])),
                "age_raw": float(age_raw_np[i]),
                "weight_raw": float(weight_raw_np[i]),
            }
            if rgb_pred_raw_np is not None:
                row["rgb_pred"] = float(rgb_pred_raw_np[i])
            if pc_pred_raw_np is not None:
                row["pc_pred"] = float(pc_pred_raw_np[i])
            if base_pred_raw_np is not None:
                row["base_pred"] = float(base_pred_raw_np[i])
            if delta_raw_np is not None:
                row["delta"] = float(delta_raw_np[i])
            if conf_np is not None:
                row["w_rgb"] = float(conf_np[i, 0])
                row["w_pc"] = float(conf_np[i, 1])
            meta_rows.append(row)

    y_true_norm_all = np.concatenate(y_true_norm_all, axis=0).reshape(-1)
    y_pred_norm_all = np.concatenate(y_pred_norm_all, axis=0).reshape(-1)
    y_true = denormalize_labels(y_true_norm_all, label_mean, label_std)
    y_pred = denormalize_labels(y_pred_norm_all, label_mean, label_std)
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        "loss": total_loss / max(total_samples, 1),
        "loss_main": total_loss_main / max(total_samples, 1),
        "loss_rgb": total_loss_rgb / max(total_samples, 1),
        "loss_pc": total_loss_pc / max(total_samples, 1),
        "loss_delta_reg": total_loss_delta_reg / max(total_samples, 1),
    })
    return metrics, meta_rows


# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser(description="DGCNN + RGB(age,weight) 融合训练")

    # 数据
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./mux")

    # 归一化
    parser.add_argument("--normalize_label", type=int, default=1)
    parser.add_argument("--normalize_age", type=int, default=1)
    parser.add_argument("--normalize_weight", type=int, default=1)
    parser.add_argument("--compute_rgb_stats", type=int, default=1)
    parser.add_argument("--rgb_stats_max_samples", type=int, default=0, help="0 表示使用全部训练图像")

    # 数据加载
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--strict_num_points", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--check_files", type=int, default=1)

    # 分支输入开关（输入屏蔽式消融）
    parser.add_argument("--use_age", type=int, default=0)
    parser.add_argument("--use_weight", type=int, default=1)

    # RGB 分支
    parser.add_argument("--rgb_model_name", type=str, default="small", choices=["small", "large"])
    parser.add_argument("--rgb_pretrained", type=int, default=1)
    parser.add_argument("--rgb_dropout", type=float, default=0.2)
    parser.add_argument("--rgb_branch_fusion_type", type=str, default="concat", choices=["concat", "residual"])
    parser.add_argument("--rgb_meta_hidden_dim", type=int, default=16)
    parser.add_argument("--rgb_meta_feat_dim", type=int, default=32)
    parser.add_argument("--rgb_fusion_hidden_dim", type=int, default=128)
    parser.add_argument("--rgb_fusion_dropout", type=float, default=0.2)
    parser.add_argument("--rgb_delta_scale", type=float, default=0.3)
    parser.add_argument("--rgb_ckpt", type=str, default="")
    parser.add_argument("--rgb_pred_is_normalized", type=int, default=1)
    parser.add_argument("--freeze_rgb_branch", type=int, default=1)

    # DGCNN 分支
    parser.add_argument("--point_normal_channel", type=int, default=0)
    parser.add_argument("--point_dropout", type=float, default=0.4)
    parser.add_argument("--point_ckpt", type=str, default="")
    parser.add_argument("--pc_pred_is_normalized", type=int, default=1)
    parser.add_argument("--freeze_point_branch", type=int, default=1)

    # 融合
    parser.add_argument("--fusion_type", type=str, default="gate_feature", choices=["gate_feature", "pred_weighted_residual"])
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--fusion_dropout", type=float, default=0.3)
    parser.add_argument("--pred_delta_scale", type=float, default=0.25)

    # 损失
    parser.add_argument("--loss_name", type=str, default="smoothl1", choices=["smoothl1", "mse", "l1"])
    parser.add_argument("--aux_rgb_weight", type=float, default=0)
    parser.add_argument("--aux_pc_weight", type=float, default=0)
    parser.add_argument("--delta_reg_weight", type=float, default=0)

    # 优化
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_scheduler", type=int, default=1)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.98)

    # 训练控制
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cpu", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval_only", type=int, default=0)

    args = parser.parse_args()
    make_dir(args.out_dir)
    set_seed(args.seed)

    device = torch.device("cpu") if str2boolint(args.use_cpu) or (not torch.cuda.is_available()) else torch.device("cuda")
    print(f"[INFO] device = {device}")

    # ---------- 训练集统计量 ----------
    label_mean, label_std = compute_scalar_stats(args.train_csv, "label")
    age_mean, age_std = compute_scalar_stats(args.train_csv, "age")
    weight_mean, weight_std = compute_scalar_stats(args.train_csv, "weight")

    rgb_stats_cache = os.path.join(args.out_dir, "train_rgb_stats.json")
    if str2boolint(args.compute_rgb_stats):
        rgb_mean, rgb_std = compute_rgb_stats(
            args.train_csv,
            image_size=(args.image_h, args.image_w),
            max_samples=args.rgb_stats_max_samples,
            cache_json=rgb_stats_cache,
        )
    else:
        rgb_mean, rgb_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    norm_stats = {
        "label_mean": float(label_mean),
        "label_std": float(label_std),
        "age_mean": float(age_mean),
        "age_std": float(age_std),
        "weight_mean": float(weight_mean),
        "weight_std": float(weight_std),
        "rgb_mean": [float(x) for x in rgb_mean],
        "rgb_std": [float(x) for x in rgb_std],
    }
    save_json(norm_stats, os.path.join(args.out_dir, "norm_stats.json"))

    print(f"[INFO] label_mean  = {label_mean:.6f}, label_std  = {label_std:.6f}")
    print(f"[INFO] age_mean    = {age_mean:.6f}, age_std    = {age_std:.6f}")
    print(f"[INFO] weight_mean = {weight_mean:.6f}, weight_std = {weight_std:.6f}")
    print(f"[INFO] rgb_mean    = {rgb_mean}")
    print(f"[INFO] rgb_std     = {rgb_std}")

    # ---------- dataset / loader ----------
    train_dataset = create_dataset(
        csv_path=args.train_csv,
        image_size=(args.image_h, args.image_w),
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
        num_points=args.num_points,
        strict_num_points=str2boolint(args.strict_num_points),
        normalize_label=str2boolint(args.normalize_label),
        label_mean=label_mean,
        label_std=label_std,
        normalize_age=str2boolint(args.normalize_age),
        age_mean=age_mean,
        age_std=age_std,
        normalize_weight=str2boolint(args.normalize_weight),
        weight_mean=weight_mean,
        weight_std=weight_std,
        check_files=str2boolint(args.check_files),
    )
    val_dataset = create_dataset(
        csv_path=args.val_csv,
        image_size=(args.image_h, args.image_w),
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
        num_points=args.num_points,
        strict_num_points=str2boolint(args.strict_num_points),
        normalize_label=str2boolint(args.normalize_label),
        label_mean=label_mean,
        label_std=label_std,
        normalize_age=str2boolint(args.normalize_age),
        age_mean=age_mean,
        age_std=age_std,
        normalize_weight=str2boolint(args.normalize_weight),
        weight_mean=weight_mean,
        weight_std=weight_std,
        check_files=str2boolint(args.check_files),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ---------- model ----------
    model = build_fusion_regression_model(
        rgb_model_name=args.rgb_model_name,
        rgb_pretrained=str2boolint(args.rgb_pretrained),
        rgb_dropout=args.rgb_dropout,
        rgb_branch_fusion_type=args.rgb_branch_fusion_type,
        rgb_meta_hidden_dim=args.rgb_meta_hidden_dim,
        rgb_meta_feat_dim=args.rgb_meta_feat_dim,
        rgb_fusion_hidden_dim=args.rgb_fusion_hidden_dim,
        rgb_fusion_dropout=args.rgb_fusion_dropout,
        rgb_delta_scale=args.rgb_delta_scale,
        use_age=str2boolint(args.use_age),
        use_weight=str2boolint(args.use_weight),
        point_normal_channel=str2boolint(args.point_normal_channel),
        point_dropout=args.point_dropout,
        fusion_type=args.fusion_type,
        proj_dim=args.proj_dim,
        fusion_dropout=args.fusion_dropout,
        pred_delta_scale=args.pred_delta_scale,
        rgb_pred_is_normalized=str2boolint(args.rgb_pred_is_normalized),
        pc_pred_is_normalized=str2boolint(args.pc_pred_is_normalized),
        label_mean=label_mean,
        label_std=label_std,
        freeze_rgb_branch=str2boolint(args.freeze_rgb_branch),
        freeze_point_branch=str2boolint(args.freeze_point_branch),
    ).to(device)

    if args.rgb_ckpt.strip() != "":
        missing, unexpected = model.load_rgb_branch_checkpoint(args.rgb_ckpt, strict=False)
        print(f"[INFO] 已加载 RGB 分支 ckpt: {args.rgb_ckpt}")
        print(f"[INFO] RGB missing={len(missing)}, unexpected={len(unexpected)}")

    if args.point_ckpt.strip() != "":
        missing, unexpected = model.load_point_branch_checkpoint(args.point_ckpt, strict=False)
        print(f"[INFO] 已加载 PointNet++ 分支 ckpt: {args.point_ckpt}")
        print(f"[INFO] DGCNN missing={len(missing)}, unexpected={len(unexpected)}")

    print(f"[INFO] total params     : {model.num_total_params():,}")
    print(f"[INFO] trainable params : {model.num_trainable_params():,}")

    criterion = FusionRegressionLoss(
        loss_type=args.loss_name,
        aux_rgb_weight=args.aux_rgb_weight,
        aux_pc_weight=args.aux_pc_weight,
        delta_reg_weight=args.delta_reg_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if str2boolint(args.use_scheduler):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    log_csv_path = os.path.join(args.out_dir, "train_log.csv")
    best_model_path = os.path.join(args.out_dir, "best_model.pth")
    last_model_path = os.path.join(args.out_dir, "last_model.pth")
    best_train_pred_path = os.path.join(args.out_dir, "train_predictions_best.csv")
    best_val_pred_path = os.path.join(args.out_dir, "val_predictions_best.csv")

    start_epoch = 1
    best_val_mae = float("inf")
    best_epoch = -1
    log_rows: List[Dict] = []

    if args.resume.strip() != "":
        ckpt = load_checkpoint(args.resume, model=model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_mae = float(ckpt.get("best_val_mae", best_val_mae))
        print(f"[INFO] resume from: {args.resume}")
        print(f"[INFO] start_epoch={start_epoch}, best_val_mae={best_val_mae:.6f}")

    if str2boolint(args.eval_only):
        ckpt_path = args.resume if args.resume.strip() != "" else best_model_path
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")
        load_checkpoint(ckpt_path, model=model, map_location=device)
        val_metrics, val_meta_rows = evaluate_one_epoch(model, val_loader, criterion, device, label_mean, label_std)
        print("=" * 100)
        print(f"[VAL] loss={val_metrics['loss']:.6f} mae={val_metrics['mae']:.4f} rmse={val_metrics['rmse']:.4f} mape={val_metrics['mape']:.2f}% r2={val_metrics['r2']:.4f}")
        print("=" * 100)
        pd.DataFrame(val_meta_rows).to_csv(best_val_pred_path, index=False, encoding="utf-8-sig")
        return

    # ---------- training ----------
    train_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, label_mean, label_std)
        val_metrics, _ = evaluate_one_epoch(model, val_loader, criterion, device, label_mean, label_std)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        total_time = time.time() - train_start

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_loss_main": train_metrics["loss_main"],
            "train_loss_rgb": train_metrics["loss_rgb"],
            "train_loss_pc": train_metrics["loss_pc"],
            "train_loss_delta_reg": train_metrics["loss_delta_reg"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_mape": train_metrics["mape"],
            "train_r2": train_metrics["r2"],
            "val_loss": val_metrics["loss"],
            "val_loss_main": val_metrics["loss_main"],
            "val_loss_rgb": val_metrics["loss_rgb"],
            "val_loss_pc": val_metrics["loss_pc"],
            "val_loss_delta_reg": val_metrics["loss_delta_reg"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_mape": val_metrics["mape"],
            "val_r2": val_metrics["r2"],
            "best_val_mae": min(best_val_mae, val_metrics["mae"]),
            "epoch_time": format_seconds(epoch_time),
            "total_time": format_seconds(total_time),
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(log_csv_path, index=False, encoding="utf-8-sig")

        is_best = val_metrics["mae"] < best_val_mae
        if is_best:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            save_checkpoint(best_model_path, epoch, model, optimizer, scheduler, best_val_mae, norm_stats, vars(args))

        save_checkpoint(last_model_path, epoch, model, optimizer, scheduler, best_val_mae, norm_stats, vars(args))

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_metrics['loss']:.6f} train_mae={train_metrics['mae']:.4f} "
            f"| val_loss={val_metrics['loss']:.6f} val_mae={val_metrics['mae']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} val_mape={val_metrics['mape']:.2f}% val_r2={val_metrics['r2']:.4f} "
            f"| best_val_mae={best_val_mae:.4f} "
            f"| epoch_time={format_seconds(epoch_time)} total_time={format_seconds(total_time)}"
        )

    # ---------- 导出 best 预测 ----------
    if os.path.isfile(best_model_path):
        load_checkpoint(best_model_path, model=model, map_location=device)
        _, train_meta_rows = evaluate_one_epoch(model, train_loader, criterion, device, label_mean, label_std)
        _, val_meta_rows = evaluate_one_epoch(model, val_loader, criterion, device, label_mean, label_std)
        pd.DataFrame(train_meta_rows).to_csv(best_train_pred_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(val_meta_rows).to_csv(best_val_pred_path, index=False, encoding="utf-8-sig")

    summary = {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "fusion_type": args.fusion_type,
        "use_age": str2boolint(args.use_age),
        "use_weight": str2boolint(args.use_weight),
        "rgb_branch_fusion_type": args.rgb_branch_fusion_type,
    }
    save_json(summary, os.path.join(args.out_dir, "summary.json"))

    print("=" * 100)
    print(f"[DONE] best_epoch={best_epoch}, best_val_mae={best_val_mae:.6f}")
    print(f"[DONE] best_model: {best_model_path}")
    print(f"[DONE] train_predictions_best: {best_train_pred_path}")
    print(f"[DONE] val_predictions_best: {best_val_pred_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
