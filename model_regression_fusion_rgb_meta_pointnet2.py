# -*- coding: utf-8 -*-
"""
model_regression_fusion_rgb_meta_pointnet2.py

最终融合模型：
1. 点云分支使用 PointNet++ SSG；
2. RGB 分支兼容现有 rgb+age+weight 单分支 ckpt；
3. age / weight 通过输入屏蔽开关做消融；
4. 支持两种融合方式：
   - gate_feature           : 原版 gate 特征融合
   - pred_weighted_residual : 预测加权 + 特征残差修正
"""

from __future__ import annotations

import os
from typing import Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    mobilenet_v3_small,
    mobilenet_v3_large,
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
)

from models.pointnet2_utils import PointNetSetAbstraction


# =========================
# checkpoint 工具
# =========================
def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    return new_state



def _extract_state_dict(ckpt: Union[str, Dict]) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, str):
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
        ckpt = torch.load(ckpt, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint 类型应为 str 或 dict，实际得到 {type(ckpt)}")

    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    return _strip_module_prefix(state_dict)


# =========================
# RGB meta 分支（兼容单分支 ckpt）
# =========================
class MetaMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 16, out_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MobileNetV3MetaBranch(nn.Module):
    """
    为了兼容你现有 rgb+age+weight 单分支 ckpt，模块命名保持与 model_regression_meta_fusion.py 对齐。

    注意：
    - meta 输入固定为 2 维 [age, weight]；
    - use_age / use_weight 只做输入屏蔽，不改网络结构，这样 best ckpt 可直接加载；
    - hidden_feat 提取：
        concat   -> regressor 最后一层前的隐藏层输出
        residual -> delta_head 最后一层前的隐藏层输出
    """
    def __init__(
        self,
        model_name: Literal["small", "large"] = "small",
        pretrained: bool = True,
        out_dim: int = 1,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        meta_dim: int = 2,
        branch_fusion_type: Literal["concat", "residual"] = "concat",
        meta_hidden_dim: int = 16,
        meta_feat_dim: int = 32,
        fusion_hidden_dim: int = 128,
        fusion_dropout: float = 0.2,
        delta_scale: float = 0.3,
        use_age: bool = True,
        use_weight: bool = True,
    ):
        super().__init__()
        if model_name == "small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
        elif model_name == "large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_large(weights=weights)
        else:
            raise ValueError(f"未知 model_name: {model_name}")

        self.meta_dim = int(meta_dim)
        self.branch_fusion_type = str(branch_fusion_type)
        self.use_age = bool(use_age)
        self.use_weight = bool(use_weight)
        self.delta_scale = float(delta_scale)

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        image_flatten_in_features = backbone.classifier[0].in_features
        image_feat_dim = backbone.classifier[0].out_features

        self.image_neck = nn.Sequential(
            nn.Linear(image_flatten_in_features, image_feat_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=float(dropout), inplace=True),
        )

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
            for p in self.avgpool.parameters():
                p.requires_grad = False
            for p in self.image_neck.parameters():
                p.requires_grad = False

        self.meta_branch = MetaMLP(
            in_dim=self.meta_dim,
            hidden_dim=meta_hidden_dim,
            out_dim=meta_feat_dim,
            dropout=fusion_dropout,
        )

        if self.branch_fusion_type == "concat":
            in_dim = image_feat_dim + meta_feat_dim
            self.regressor = nn.Sequential(
                nn.Linear(in_dim, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(fusion_dropout)),
                nn.Linear(fusion_hidden_dim, out_dim),
            )
            self.img_head = None
            self.delta_head = None
            self.feature_dim = fusion_hidden_dim
        elif self.branch_fusion_type == "residual":
            self.img_head = nn.Sequential(
                nn.Linear(image_feat_dim, fusion_hidden_dim),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=float(fusion_dropout)),
                nn.Linear(fusion_hidden_dim, out_dim),
            )
            delta_hidden_dim = max(fusion_hidden_dim // 2, 32)
            self.delta_head = nn.Sequential(
                nn.Linear(image_feat_dim + meta_feat_dim, delta_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(fusion_dropout * 0.5)),
                nn.Linear(delta_hidden_dim, out_dim),
            )
            self.regressor = None
            self.feature_dim = delta_hidden_dim
        else:
            raise ValueError(f"未知 branch_fusion_type: {branch_fusion_type}")

        self.image_feat_dim = image_feat_dim
        self.meta_feat_dim = meta_feat_dim

    def extract_image_feature(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.image_neck(x)
        return x

    def _prepare_meta(self, meta: Optional[torch.Tensor], batch_size: int, dtype, device) -> torch.Tensor:
        if meta is None:
            meta = torch.zeros(batch_size, 2, dtype=dtype, device=device)
        if meta.ndim == 1:
            meta = meta.unsqueeze(0)
        if meta.ndim != 2 or meta.shape[1] != 2:
            raise ValueError(f"meta 形状应为 [B,2]，实际为 {tuple(meta.shape)}")
        meta = meta.to(device=device, dtype=dtype)

        # 输入屏蔽式消融，保持结构不变，兼容 rgb+age+weight 单分支 ckpt
        if not self.use_age:
            meta[:, 0] = 0.0
        if not self.use_weight:
            meta[:, 1] = 0.0
        return meta

    def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None):
        img_feat = self.extract_image_feature(x)
        meta_in = self._prepare_meta(meta, img_feat.shape[0], img_feat.dtype, img_feat.device)
        meta_feat = self.meta_branch(meta_in)
        fused_input = torch.cat([img_feat, meta_feat], dim=1)

        if self.branch_fusion_type == "concat":
            hidden_feat = self.regressor[0](fused_input)
            hidden_feat = self.regressor[1](hidden_feat)
            hidden_feat = self.regressor[2](hidden_feat)
            pred = self.regressor[3](hidden_feat)
            pred_base = None
            delta = None
        else:
            pred_base = self.img_head(img_feat)
            hidden_feat = self.delta_head[0](fused_input)
            hidden_feat = self.delta_head[1](hidden_feat)
            hidden_feat = self.delta_head[2](hidden_feat)
            delta = self.delta_scale * torch.tanh(self.delta_head[3](hidden_feat))
            pred = pred_base + delta

        aux = {
            "img_feat": img_feat,
            "meta_feat": meta_feat,
            "fused_input": fused_input,
            "pred_base": pred_base,
            "delta": delta,
            "meta_used": meta_in,
        }
        return pred, hidden_feat, aux


# =========================
# PointNet++ 分支（兼容单分支 ckpt）
# =========================
class PointNetSSGBranch(nn.Module):
    """
    严格沿用你之前 PointNet++ 融合版的单分支结构：
    - hidden feat 取自 fc2 + bn2 + relu + dropout 后的 512 维
    - 最终回归头为 fc3(512->1)
    """
    def __init__(self, normal_channel: bool = False, dropout: float = 0.4):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = bool(normal_channel)

        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel, mlp=[64, 128, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 256, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 2048], group_all=True
        )

        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(float(dropout))

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(float(dropout))

        self.fc3 = nn.Linear(512, 1)
        self.feature_dim = 512

    @staticmethod
    def _ensure_bcn(points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 3:
            raise ValueError(f"points 必须是 3D tensor，实际得到 shape={tuple(points.shape)}")

        _, d1, d2 = points.shape
        if d1 in (3, 6):
            return points
        if d2 in (3, 6):
            return points.permute(0, 2, 1).contiguous()
        raise ValueError(
            f"无法判断 points 的通道维。当前 shape={tuple(points.shape)}，"
            f"期望为 (B,N,3)/(B,3,N)/(B,N,6)/(B,6,N) 之一。"
        )

    def forward_features(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = self._ensure_bcn(points)
        b, c, _ = xyz.shape

        if self.normal_channel:
            if c != 6:
                raise ValueError(f"normal_channel=True 时输入通道数应为 6，实际 C={c}")
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            if c != 3:
                raise ValueError(f"normal_channel=False 时输入通道数应为 3，实际 C={c}")
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(b, 2048)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        feat = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return feat, l3_points

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, global_feat = self.forward_features(points)
        pred = self.fc3(feat)
        return pred, feat, global_feat


# =========================
# 融合模型
# =========================
class RGBMetaPointNet2FusionRegressor(nn.Module):
    def __init__(
        self,
        # RGB 分支
        rgb_model_name: Literal["small", "large"] = "small",
        rgb_pretrained: bool = True,
        rgb_dropout: float = 0.2,
        rgb_branch_fusion_type: Literal["concat", "residual"] = "concat",
        rgb_meta_hidden_dim: int = 16,
        rgb_meta_feat_dim: int = 32,
        rgb_fusion_hidden_dim: int = 128,
        rgb_fusion_dropout: float = 0.2,
        rgb_delta_scale: float = 0.3,
        use_age: bool = True,
        use_weight: bool = True,
        # PointNet++ 分支
        point_normal_channel: bool = False,
        point_dropout: float = 0.4,
        # 融合
        fusion_type: Literal["gate_feature", "pred_weighted_residual"] = "pred_weighted_residual",
        proj_dim: int = 256,
        fusion_dropout: float = 0.3,
        pred_delta_scale: float = 0.25,
        # 预测空间兼容
        rgb_pred_is_normalized: bool = True,
        pc_pred_is_normalized: bool = True,
        label_mean: float = 0.0,
        label_std: float = 1.0,
        # 冻结
        freeze_rgb_branch: bool = False,
        freeze_point_branch: bool = False,
    ):
        super().__init__()
        self.fusion_type = str(fusion_type)
        self.rgb_pred_is_normalized = bool(rgb_pred_is_normalized)
        self.pc_pred_is_normalized = bool(pc_pred_is_normalized)
        self.label_mean = float(label_mean)
        self.label_std = float(label_std) if abs(float(label_std)) > 1e-12 else 1.0
        self.freeze_rgb_branch = bool(freeze_rgb_branch)
        self.freeze_point_branch = bool(freeze_point_branch)
        self.pred_delta_scale = float(pred_delta_scale)

        self.rgb_branch = MobileNetV3MetaBranch(
            model_name=rgb_model_name,
            pretrained=rgb_pretrained,
            out_dim=1,
            dropout=rgb_dropout,
            freeze_backbone=False,
            meta_dim=2,
            branch_fusion_type=rgb_branch_fusion_type,
            meta_hidden_dim=rgb_meta_hidden_dim,
            meta_feat_dim=rgb_meta_feat_dim,
            fusion_hidden_dim=rgb_fusion_hidden_dim,
            fusion_dropout=rgb_fusion_dropout,
            delta_scale=rgb_delta_scale,
            use_age=use_age,
            use_weight=use_weight,
        )
        self.point_branch = PointNetSSGBranch(
            normal_channel=point_normal_channel,
            dropout=point_dropout,
        )

        self.rgb_proj = nn.Sequential(
            nn.Linear(self.rgb_branch.feature_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(fusion_dropout)),
        )
        self.pc_proj = nn.Sequential(
            nn.Linear(self.point_branch.feature_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(fusion_dropout)),
        )

        if self.fusion_type == "gate_feature":
            self.gate = nn.Sequential(
                nn.Linear(proj_dim * 2, proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim),
                nn.Sigmoid(),
            )
            self.fusion_head = nn.Sequential(
                nn.Linear(proj_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(float(fusion_dropout)),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(float(fusion_dropout)),
                nn.Linear(64, 1),
            )
            self.conf_head = None
            self.delta_head = None
        elif self.fusion_type == "pred_weighted_residual":
            inter_dim = proj_dim * 4
            self.gate = None
            self.conf_head = nn.Sequential(
                nn.Linear(inter_dim + 3, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(float(fusion_dropout * 0.5)),
                nn.Linear(128, 2),
            )
            self.delta_head = nn.Sequential(
                nn.Linear(inter_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(float(fusion_dropout)),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(float(fusion_dropout * 0.5)),
                nn.Linear(64, 1),
            )
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
            self.fusion_head = None
        else:
            raise ValueError(f"未知 fusion_type: {fusion_type}")

        if self.freeze_rgb_branch:
            for p in self.rgb_branch.parameters():
                p.requires_grad = False
        if self.freeze_point_branch:
            for p in self.point_branch.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_rgb_branch:
            self.rgb_branch.eval()
        if mode and self.freeze_point_branch:
            self.point_branch.eval()
        return self

    def _to_norm_space(self, pred: torch.Tensor, already_norm: bool) -> torch.Tensor:
        if already_norm:
            return pred
        return (pred - self.label_mean) / self.label_std

    def load_rgb_branch_checkpoint(self, ckpt: Union[str, Dict], strict: bool = False):
        state_dict = _extract_state_dict(ckpt)
        missing, unexpected = self.rgb_branch.load_state_dict(state_dict, strict=strict)
        return missing, unexpected

    def load_point_branch_checkpoint(self, ckpt: Union[str, Dict], strict: bool = False):
        state_dict = _extract_state_dict(ckpt)
        missing, unexpected = self.point_branch.load_state_dict(state_dict, strict=strict)
        return missing, unexpected

    def forward(self, image: torch.Tensor, points: torch.Tensor, meta: Optional[torch.Tensor] = None):
        rgb_pred_raw, rgb_feat_raw, rgb_aux = self.rgb_branch(image, meta)
        pc_pred_raw, pc_feat_raw, pc_global_feat = self.point_branch(points)

        rgb_pred_norm = self._to_norm_space(rgb_pred_raw, self.rgb_pred_is_normalized)
        pc_pred_norm = self._to_norm_space(pc_pred_raw, self.pc_pred_is_normalized)

        z_rgb = self.rgb_proj(rgb_feat_raw)
        z_pc = self.pc_proj(pc_feat_raw)

        if self.fusion_type == "gate_feature":
            gate = self.gate(torch.cat([z_rgb, z_pc], dim=1))
            fused_feat = gate * z_rgb + (1.0 - gate) * z_pc
            fused_pred = self.fusion_head(fused_feat)
            base_pred = None
            delta = None
            conf = None
        else:
            inter_feat = torch.cat([z_rgb, z_pc, torch.abs(z_rgb - z_pc), z_rgb * z_pc], dim=1)
            pred_gap = torch.abs(rgb_pred_norm - pc_pred_norm)
            conf_in = torch.cat([inter_feat, rgb_pred_norm, pc_pred_norm, pred_gap], dim=1)
            conf_logits = self.conf_head(conf_in)
            conf = torch.softmax(conf_logits, dim=1)
            w_rgb = conf[:, 0:1]
            w_pc = conf[:, 1:2]
            base_pred = w_rgb * rgb_pred_norm + w_pc * pc_pred_norm
            delta = self.pred_delta_scale * torch.tanh(self.delta_head(inter_feat))
            fused_pred = base_pred + delta
            fused_feat = inter_feat
            gate = None

        aux = {
            "rgb_pred_raw": rgb_pred_raw,
            "pc_pred_raw": pc_pred_raw,
            "rgb_pred": rgb_pred_norm,
            "pc_pred": pc_pred_norm,
            "rgb_feat_raw": rgb_feat_raw,
            "pc_feat_raw": pc_feat_raw,
            "rgb_feat_proj": z_rgb,
            "pc_feat_proj": z_pc,
            "pc_global_feat": pc_global_feat,
            "fused_feat": fused_feat,
            "gate": gate,
            "conf": conf,
            "base_pred": base_pred,
            "delta": delta,
        }
        aux.update(rgb_aux)
        return fused_pred, aux

    def forward_batch(self, batch: Dict[str, torch.Tensor]):
        return self.forward(image=batch["image"], points=batch["points"], meta=batch.get("meta", None))

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =========================
# 损失函数
# =========================
class FusionRegressionLoss(nn.Module):
    def __init__(
        self,
        loss_type: Literal["smoothl1", "mse", "l1"] = "smoothl1",
        aux_rgb_weight: float = 0.0,
        aux_pc_weight: float = 0.0,
        delta_reg_weight: float = 0.0,
    ):
        super().__init__()
        self.loss_type = str(loss_type).lower().strip()
        self.aux_rgb_weight = float(aux_rgb_weight)
        self.aux_pc_weight = float(aux_pc_weight)
        self.delta_reg_weight = float(delta_reg_weight)
        if self.loss_type not in {"smoothl1", "mse", "l1"}:
            raise ValueError(f"未知 loss_type: {loss_type}")

    def _base_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float().view(-1, 1)
        target = target.float().view(-1, 1)
        if self.loss_type == "smoothl1":
            return F.smooth_l1_loss(pred, target)
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        return F.l1_loss(pred, target)

    def forward(self, fused_pred: torch.Tensor, target: torch.Tensor, aux: Dict[str, torch.Tensor]):
        loss_main = self._base_loss(fused_pred, target)
        total = loss_main

        loss_rgb = torch.tensor(0.0, device=fused_pred.device)
        if self.aux_rgb_weight > 0.0 and aux.get("rgb_pred", None) is not None:
            loss_rgb = self._base_loss(aux["rgb_pred"], target)
            total = total + self.aux_rgb_weight * loss_rgb

        loss_pc = torch.tensor(0.0, device=fused_pred.device)
        if self.aux_pc_weight > 0.0 and aux.get("pc_pred", None) is not None:
            loss_pc = self._base_loss(aux["pc_pred"], target)
            total = total + self.aux_pc_weight * loss_pc

        loss_delta_reg = torch.tensor(0.0, device=fused_pred.device)
        if self.delta_reg_weight > 0.0 and aux.get("delta", None) is not None:
            loss_delta_reg = aux["delta"].abs().mean()
            total = total + self.delta_reg_weight * loss_delta_reg

        detail = {
            "loss": total.detach(),
            "loss_main": loss_main.detach(),
            "loss_rgb": loss_rgb.detach(),
            "loss_pc": loss_pc.detach(),
            "loss_delta_reg": loss_delta_reg.detach(),
        }
        return total, detail



def build_fusion_regression_model(**kwargs):
    return RGBMetaPointNet2FusionRegressor(**kwargs)


if __name__ == "__main__":
    pass
