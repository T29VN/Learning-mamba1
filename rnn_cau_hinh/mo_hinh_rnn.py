"""Elman RNN đối chứng cho bài toán phân loại hướng sóng đến (DOA)."""

import torch
import torch.nn as nn

from . import cau_hinh_rnn as cfg


# ============================================================
# 1. ĐỊNH NGHĨA MODEL
# ============================================================

class ElmanRNNDOAClassifier(nn.Module):
    """Phân loại DOA từ chuỗi tín hiệu I/Q bằng Elman RNN."""

    # ========================================================
    # 2. CONSTRUCTOR
    # ========================================================

    def __init__(self):
        super().__init__()

        self._validate_configuration()

        layer_input_sizes = [cfg.INPUT_SIZE] + [
            cfg.HIDDEN_SIZE
            for _ in range(cfg.NUM_LAYERS - 1)
        ]

        self.pre_norms = nn.ModuleList(
            [nn.LayerNorm(input_size) for input_size in layer_input_sizes]
        )
        self.rnn_layers = nn.ModuleList(
            [
                nn.RNN(
                    input_size=input_size,
                    hidden_size=cfg.HIDDEN_SIZE,
                    num_layers=1,
                    nonlinearity=cfg.NONLINEARITY,
                    bias=cfg.BIAS,
                    batch_first=True,
                    dropout=0.0,
                    bidirectional=cfg.BIDIRECTIONAL,
                )
                for input_size in layer_input_sizes
            ]
        )

        self.final_norm = (
            nn.LayerNorm(cfg.HIDDEN_SIZE)
            if cfg.USE_FINAL_NORM
            else nn.Identity()
        )
        self.dropout = nn.Dropout(cfg.DROPOUT)
        self.classifier = nn.Linear(cfg.HIDDEN_SIZE, cfg.NUM_CLASSES)

    @staticmethod
    def _validate_configuration() -> None:
        """Kiểm tra config vẫn khớp kiến trúc RNN đã chốt."""
        if cfg.RNN_LAYER_MODE != "separate_single_layer":
            raise ValueError(
                "ElmanRNNDOAClassifier chỉ hỗ trợ "
                "RNN_LAYER_MODE='separate_single_layer', "
                f"nhưng nhận được {cfg.RNN_LAYER_MODE!r}."
            )
        if not cfg.USE_LAYER_NORM:
            raise ValueError("ElmanRNNDOAClassifier yêu cầu USE_LAYER_NORM=True.")
        if cfg.NORM_POSITION != "pre":
            raise ValueError(
                "ElmanRNNDOAClassifier chỉ hỗ trợ NORM_POSITION='pre', "
                f"nhưng nhận được {cfg.NORM_POSITION!r}."
            )
        if cfg.NONLINEARITY != "tanh":
            raise ValueError(
                "ElmanRNNDOAClassifier chỉ hỗ trợ NONLINEARITY='tanh', "
                f"nhưng nhận được {cfg.NONLINEARITY!r}."
            )
        if cfg.BIDIRECTIONAL:
            raise ValueError(
                "ElmanRNNDOAClassifier hiện chỉ hỗ trợ RNN unidirectional."
            )
        if cfg.USE_RESIDUAL:
            raise ValueError(
                "ElmanRNNDOAClassifier baseline không hỗ trợ residual."
            )
        if cfg.LEARNABLE_H0:
            raise ValueError(
                "ElmanRNNDOAClassifier baseline không hỗ trợ learnable h0."
            )
        if cfg.POOLING_TYPE != "mean":
            raise ValueError(
                "ElmanRNNDOAClassifier chỉ hỗ trợ POOLING_TYPE='mean', "
                f"nhưng nhận được {cfg.POOLING_TYPE!r}."
            )
        if cfg.NUM_LAYERS < 1:
            raise ValueError("NUM_LAYERS phải lớn hơn hoặc bằng 1.")

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
            cfg.INPUT_SIZE,
        ).contiguous()

        for norm, rnn in zip(self.pre_norms, self.rnn_layers):
            x = norm(x)
            x, _ = rnn(x)

        x = self.final_norm(x)

        pooled_features = x.mean(dim=1)
        classifier_input = self.dropout(pooled_features)
        logits = self.classifier(classifier_input)

        return {
            "logits": logits,
            "pooled_features": pooled_features,
        }
