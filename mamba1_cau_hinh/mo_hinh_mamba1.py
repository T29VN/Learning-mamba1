"""Mô hình Mamba-1 cho bài toán phân loại hướng sóng đến (DOA)."""

import torch
import torch.nn as nn

from mamba_ssm import Mamba

from . import cau_hinh_mamba1 as cfg


# ============================================================
# 1. ĐỊNH NGHĨA MODEL
# ============================================================

class Mamba1DOAClassifier(nn.Module):
    """Phân loại DOA từ chuỗi tín hiệu I/Q thu bởi dàn anten."""

    # ========================================================
    # 2. CONSTRUCTOR
    # ========================================================

    def __init__(self):
        super().__init__()

        # cái này để kiểm tra cho chắc chắn là đúng cấu hình không
        # nếu sai thì sẽ báo lỗi ngay từ đầu
        if cfg.NORM_POSITION != "pre":
            raise ValueError(
                "Mamba1DOAClassifier chỉ hỗ trợ NORM_POSITION='pre', "
                f"nhưng nhận được {cfg.NORM_POSITION!r}."
            )
        
        # cái này để kiểm tra cho chắc chắn là đúng cấu hình không
        # nếu sai thì sẽ báo lỗi ngay từ đầu
        if cfg.POOLING_TYPE != "mean":
            raise ValueError(
                "Mamba1DOAClassifier chỉ hỗ trợ POOLING_TYPE='mean', "
                f"nhưng nhận được {cfg.POOLING_TYPE!r}."
            )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(cfg.D_MODEL) for _ in range(cfg.NUM_LAYERS)]
        )

        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=cfg.D_MODEL,
                    d_state=cfg.D_STATE,
                    d_conv=cfg.D_CONV,
                    expand=cfg.EXPAND,
                )
                for _ in range(cfg.NUM_LAYERS) # dùng for vì dưới dùng 3 khối mamba
            ]
        )

        self.final_norm = (
            nn.LayerNorm(cfg.D_MODEL)
            if cfg.USE_FINAL_NORM
            else nn.Identity()
        )
        self.dropout = nn.Dropout(cfg.DROPOUT)
        self.classifier = nn.Linear(cfg.D_MODEL, cfg.NUM_CLASSES)

    # ========================================================
    # 3. KIỂM TRA INPUT
    # ========================================================

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        expected_shape = (
            "[B, "
            f"{cfg.SEQUENCE_LENGTH}, {cfg.NUM_IQ}, {cfg.NUM_ANTENNAS}]"
        )
        received_shape = list(x.shape)

        if x.ndim != 4:
            raise ValueError(
                f"Input phải có shape {expected_shape}, "
                f"nhưng nhận được {received_shape}."
            )

        if (
            x.shape[1] != cfg.SEQUENCE_LENGTH
            or x.shape[2] != cfg.NUM_IQ
            or x.shape[3] != cfg.NUM_ANTENNAS
        ):
            raise ValueError(
                f"Input phải có shape {expected_shape}, "
                f"nhưng nhận được {received_shape}."
            )

    # ========================================================
    # 4. FORWARD
    # ========================================================

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(x)

        batch_size = x.shape[0]
        x = x.reshape(
            batch_size,
            cfg.SEQUENCE_LENGTH,
            cfg.D_MODEL,
        ).contiguous()

        for norm, mamba in zip(self.norms, self.mamba_layers):
            transformed = mamba(norm(x))

            if cfg.USE_RESIDUAL:
                x = x + transformed
            else:
                x = transformed

        x = self.final_norm(x)

        pooled_features = x.mean(dim=1)
        classifier_input = self.dropout(pooled_features)
        logits = self.classifier(classifier_input)

        return {
            "logits": logits,
            "pooled_features": pooled_features,
        }


