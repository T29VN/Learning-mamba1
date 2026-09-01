"""Huấn luyện, validation và test mô hình Mamba-1 phân loại DOA."""

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm.auto import tqdm
except ImportError as exc:
    raise ImportError(
        "Thiếu dependency 'tqdm'. Hãy cài tqdm trước khi chạy training."
    ) from exc

from mamba1_cau_hinh import cau_hinh_mamba1 as cfg
from mamba1_cau_hinh.mo_hinh_mamba1 import Mamba1DOAClassifier


REQUIRED_SPLIT_KEYS = (
    "signals",
    "doa_class",
    "doa_degree",
    "snr_db",
)

HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_rmse_degree",
    "learning_rate",
)

OUTPUT_FILENAMES = {
    "best_model": "best_model.pt",
    "last_checkpoint": "last_checkpoint.pt",
    "history": "history.csv",
    "test_by_snr": "test_by_snr.csv",
    "test_summary": "test_summary.txt",
}


# ============================================================
# 1. CẤU HÌNH RUNTIME VÀ SEED
# ============================================================

def validate_training_configuration() -> None:
    """Kiểm tra các lựa chọn mà baseline training hiện tại hỗ trợ."""
    expected_num_classes = (
        (cfg.DOA_MAX_DEGREE - cfg.DOA_MIN_DEGREE)
        // cfg.DOA_STEP_DEGREE
        + 1
    )
    if expected_num_classes != cfg.NUM_CLASSES:
        raise ValueError(
            "NUM_CLASSES không khớp với DOA_MIN_DEGREE, "
            "DOA_MAX_DEGREE và DOA_STEP_DEGREE."
        )

    if cfg.USE_AMP is not True:
        raise ValueError("Baseline này yêu cầu USE_AMP = True.")

    if cfg.LR_SCHEDULER is not None:
        raise ValueError("Baseline này chưa hỗ trợ learning-rate scheduler.")

    if cfg.GRAD_CLIP is not None and cfg.GRAD_CLIP <= 0:
        raise ValueError("GRAD_CLIP phải là số dương hoặc None.")


def get_cuda_device() -> torch.device:
    """Lấy CUDA device bắt buộc cho baseline Mamba-1 có AMP."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Baseline Mamba-1 yêu cầu CUDA, nhưng torch.cuda.is_available() "
            "đang trả về False."
        )
    return torch.device("cuda")


def set_seed() -> None:
    """Đặt seed cho các nguồn sinh số ngẫu nhiên được sử dụng."""
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)


# ============================================================
# 2. DATASET VÀ DATALOADER
# ============================================================

def validate_split(data: dict, split_name: str) -> None:
    """Kiểm tra cấu trúc, shape và mapping nhãn của một split."""
    if not isinstance(data, dict):
        raise ValueError(
            f"Split {split_name!r} phải là dictionary, "
            f"nhưng nhận được {type(data).__name__}."
        )

    missing_keys = [key for key in REQUIRED_SPLIT_KEYS if key not in data]
    if missing_keys:
        raise ValueError(
            f"Split {split_name!r} thiếu các key bắt buộc: {missing_keys}."
        )

    for key in REQUIRED_SPLIT_KEYS:
        if not torch.is_tensor(data[key]):
            raise ValueError(
                f"Split {split_name!r}: {key!r} phải là torch.Tensor."
            )

    signals = data["signals"]
    doa_class = data["doa_class"]
    doa_degree = data["doa_degree"]
    snr_db = data["snr_db"]

    expected_signal_shape = (
        cfg.SEQUENCE_LENGTH,
        cfg.NUM_IQ,
        cfg.NUM_ANTENNAS,
    )
    if signals.ndim != 4 or tuple(signals.shape[1:]) != expected_signal_shape:
        raise ValueError(
            f"Split {split_name!r}: signals phải có shape "
            f"[N, {cfg.SEQUENCE_LENGTH}, {cfg.NUM_IQ}, {cfg.NUM_ANTENNAS}], "
            f"nhưng nhận được {list(signals.shape)}."
        )

    if signals.dtype != torch.float32:
        raise ValueError(
            f"Split {split_name!r}: signals phải có dtype torch.float32, "
            f"nhưng nhận được {signals.dtype}."
        )
    if doa_class.dtype != torch.long:
        raise ValueError(
            f"Split {split_name!r}: doa_class phải có dtype torch.long, "
            f"nhưng nhận được {doa_class.dtype}."
        )
    if doa_degree.dtype != torch.float32 or snr_db.dtype != torch.float32:
        raise ValueError(
            f"Split {split_name!r}: doa_degree và snr_db phải có "
            "dtype torch.float32."
        )

    num_samples = signals.shape[0]
    if num_samples == 0:
        raise ValueError(f"Split {split_name!r} không có sample nào.")

    for key, tensor in (
        ("doa_class", doa_class),
        ("doa_degree", doa_degree),
        ("snr_db", snr_db),
    ):
        if tensor.ndim != 1:
            raise ValueError(
                f"Split {split_name!r}: {key} phải có shape [N], "
                f"nhưng nhận được {list(tensor.shape)}."
            )
        if tensor.shape[0] != num_samples:
            raise ValueError(
                f"Split {split_name!r}: {key} có {tensor.shape[0]} sample, "
                f"không khớp signals có {num_samples} sample."
            )

    min_class = int(doa_class.min().item())
    max_class = int(doa_class.max().item())
    if min_class < 0 or max_class >= cfg.NUM_CLASSES:
        raise ValueError(
            f"Split {split_name!r}: doa_class phải nằm trong "
            f"[0, {cfg.NUM_CLASSES - 1}], nhưng nhận được range "
            f"[{min_class}, {max_class}]."
        )

    expected_degree = (
        cfg.DOA_MIN_DEGREE
        + doa_class.to(dtype=doa_degree.dtype) * cfg.DOA_STEP_DEGREE
    )
    if not torch.equal(expected_degree, doa_degree):
        raise ValueError(
            f"Split {split_name!r}: doa_degree không khớp mapping "
            "DOA_MIN_DEGREE + doa_class * DOA_STEP_DEGREE."
        )


def load_split(path: Path, split_name: str) -> TensorDataset:
    """Load toàn bộ một split vào RAM và tạo TensorDataset."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy dataset split {split_name!r}: {path}"
        )

    data = torch.load(path, map_location="cpu")
    validate_split(data, split_name)

    return TensorDataset(
        data["signals"],
        data["doa_class"],
        data["doa_degree"],
        data["snr_db"],
    )


def create_dataloaders(
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, bool]:
    """Tạo DataLoader train, validation và test theo config."""
    train_dataset = load_split(cfg.TRAIN_FILE, "train")
    val_dataset = load_split(cfg.VAL_FILE, "validation")
    test_dataset = load_split(cfg.TEST_FILE, "test")

    use_pin_memory = cfg.PIN_MEMORY and device.type == "cuda"
    common_options = {
        "batch_size": cfg.BATCH_SIZE,
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": use_pin_memory,
        "drop_last": cfg.DROP_LAST,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=cfg.SHUFFLE_TRAIN,
        **common_options,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=cfg.SHUFFLE_VAL,
        **common_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=cfg.SHUFFLE_TEST,
        **common_options,
    )

    return train_loader, val_loader, test_loader, use_pin_memory


# ============================================================
# 3. MODEL, PARAMETER, OPTIMIZER VÀ LOSS
# ============================================================

def count_model_parameters(model: nn.Module) -> dict[str, int]:
    """Đếm tổng parameter và breakdown theo thành phần của model."""
    # Một parameter là một giá trị vô hướng nằm trong một tensor nn.Parameter.
    # p.numel() trả về số giá trị vô hướng trong tensor parameter p.
    counts = {
        "mamba_blocks": sum(p.numel() for p in model.mamba_layers.parameters()),
        "pre_layer_norms": sum(p.numel() for p in model.norms.parameters()),
        "final_layer_norm": sum(p.numel() for p in model.final_norm.parameters()),
        "classifier": sum(p.numel() for p in model.classifier.parameters()),
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }

    breakdown_total = (
        counts["mamba_blocks"]
        + counts["pre_layer_norms"]
        + counts["final_layer_norm"]
        + counts["classifier"]
    )
    if breakdown_total != counts["total"]:
        raise RuntimeError(
            "Parameter breakdown không bằng total parameters; "
            "có thành phần model chưa được tính."
        )

    return counts


def print_parameter_counts(counts: dict[str, int]) -> None:
    """In định nghĩa và breakdown số parameter ra Terminal."""
    print("\n===== MODEL PARAMETERS =====")
    print("Definition: 1 parameter = 1 scalar value stored in nn.Parameter.\n")
    print(f"Mamba blocks:         {counts['mamba_blocks']:,}")
    print(f"Pre-LayerNorms:       {counts['pre_layer_norms']:,}")
    print(f"Final LayerNorm:      {counts['final_layer_norm']:,}")
    print(f"Classifier:           {counts['classifier']:,}\n")
    print(f"Total parameters:     {counts['total']:,}")
    print(f"Trainable parameters: {counts['trainable']:,}")
    print("============================\n")


def create_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Tạo AdamW optimizer theo cấu hình baseline."""
    if cfg.OPTIMIZER_NAME != "AdamW":
        raise ValueError(
            "Baseline này chỉ hỗ trợ OPTIMIZER_NAME='AdamW', "
            f"nhưng nhận được {cfg.OPTIMIZER_NAME!r}."
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )


def create_loss() -> nn.Module:
    """Tạo CrossEntropyLoss dùng trực tiếp với raw logits."""
    if cfg.LOSS_NAME != "CrossEntropyLoss":
        raise ValueError(
            "Baseline này chỉ hỗ trợ LOSS_NAME='CrossEntropyLoss', "
            f"nhưng nhận được {cfg.LOSS_NAME!r}."
        )
    return nn.CrossEntropyLoss()


# ============================================================
# 4. TRAIN VÀ VALIDATION
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_pin_memory: bool,
    epoch: int,
) -> dict[str, float]:
    """Huấn luyện một epoch và trả loss/accuracy theo sample."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} - Train")
    for signals, doa_class, _doa_degree, _snr_db in progress:
        signals = signals.to(
            device,
            non_blocking=use_pin_memory,
        )
        doa_class = doa_class.to(
            device,
            non_blocking=use_pin_memory,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            output = model(signals)
            logits = output["logits"]
            loss = criterion(logits, doa_class)

        scaler.scale(loss).backward()

        if cfg.GRAD_CLIP is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        batch_size = signals.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == doa_class).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Train DataLoader không có sample nào.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * total_correct / total_samples,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_pin_memory: bool,
    description: str,
) -> dict[str, float]:
    """Tính loss, accuracy và RMSE degree trên toàn bộ một split."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_squared_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for signals, doa_class, doa_degree, _snr_db in tqdm(
            loader,
            desc=description,
        ):
            signals = signals.to(
                device,
                non_blocking=use_pin_memory,
            )
            doa_class = doa_class.to(
                device,
                non_blocking=use_pin_memory,
            )
            doa_degree = doa_degree.to(
                device,
                non_blocking=use_pin_memory,
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                output = model(signals)
                logits = output["logits"]
                loss = criterion(logits, doa_class)

            pred_class = logits.argmax(dim=1)
            pred_degree = (
                cfg.DOA_MIN_DEGREE
                + pred_class.to(dtype=doa_degree.dtype)
                * cfg.DOA_STEP_DEGREE
            )
            squared_error = (pred_degree - doa_degree).square()

            batch_size = signals.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (pred_class == doa_class).sum().item()
            total_squared_error += squared_error.sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Evaluation DataLoader không có sample nào.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * total_correct / total_samples,
        "rmse": math.sqrt(total_squared_error / total_samples),
    }


# ============================================================
# 5. OUTPUT VÀ CHECKPOINT
# ============================================================

def get_output_paths() -> dict[str, Path]:
    """Tạo mapping tên logic sang đường dẫn output trong model directory."""
    return {
        key: cfg.MODEL_SAVE_DIR / filename
        for key, filename in OUTPUT_FILENAMES.items()
    }


def prepare_output_directory(output_paths: dict[str, Path]) -> None:
    """Áp dụng chính sách chống ghi đè cho fresh training/resume."""
    if cfg.RESUME_TRAINING:
        required_resume_files = (
            output_paths["last_checkpoint"],
            output_paths["best_model"],
            output_paths["history"],
        )
        missing = [path for path in required_resume_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Không thể resume vì thiếu file: "
                + ", ".join(str(path) for path in missing)
            )
        return

    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Fresh training không được ghi đè các file đã tồn tại: "
            + ", ".join(str(path) for path in existing)
        )

    cfg.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def initialize_history(path: Path) -> None:
    """Tạo history.csv mới và ghi header."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()


def append_history(
    path: Path,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    learning_rate: float,
) -> None:
    """Append đúng một dòng metric của một epoch vào history.csv."""
    row = {
        "epoch": epoch,
        "train_loss": f"{train_metrics['loss']:.8f}",
        "train_accuracy": f"{train_metrics['accuracy']:.6f}",
        "val_loss": f"{val_metrics['loss']:.8f}",
        "val_accuracy": f"{val_metrics['accuracy']:.6f}",
        "val_rmse_degree": f"{val_metrics['rmse']:.8f}",
        "learning_rate": f"{learning_rate:.12g}",
    }
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writerow(row)


def validate_resume_history(path: Path, checkpoint_epoch: int) -> None:
    """Đảm bảo history hiện tại kết thúc đúng tại epoch checkpoint."""
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(HISTORY_FIELDS):
            raise ValueError(
                f"Header của history.csv không hợp lệ: {reader.fieldnames}."
            )
        rows = list(reader)

    if not rows:
        raise ValueError("history.csv không có dữ liệu epoch để resume.")

    try:
        history_last_epoch = int(rows[-1]["epoch"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Epoch cuối trong history.csv không hợp lệ.") from exc

    if history_last_epoch != checkpoint_epoch:
        raise ValueError(
            f"Epoch cuối của history.csv là {history_last_epoch}, "
            f"nhưng last checkpoint là epoch {checkpoint_epoch}."
        )


def build_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_rmse: float,
    best_epoch: int,
) -> dict:
    """Tạo checkpoint đầy đủ cho best model và last checkpoint."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_rmse": val_metrics["rmse"],
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
    }


def load_resume_checkpoint(
    path: Path,
    history_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, int]:
    """Khôi phục đầy đủ trạng thái và trả epoch bắt đầu tiếp theo."""
    checkpoint = torch.load(path, map_location=device)
    required_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "best_val_rmse",
        "best_epoch",
    }
    missing_keys = sorted(required_keys - checkpoint.keys())
    if missing_keys:
        raise ValueError(
            f"Last checkpoint thiếu các key bắt buộc: {missing_keys}."
        )

    last_epoch = int(checkpoint["epoch"])
    if last_epoch >= cfg.EPOCHS:
        raise ValueError(
            f"Training đã đạt epoch {last_epoch}, bằng hoặc vượt "
            f"EPOCHS={cfg.EPOCHS}; không có epoch nào để resume."
        )

    validate_resume_history(history_path, last_epoch)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return (
        last_epoch + 1,
        float(checkpoint["best_val_rmse"]),
        int(checkpoint["best_epoch"]),
    )


# ============================================================
# 6. TEST VÀ BÁO CÁO
# ============================================================

def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_pin_memory: bool,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Test đúng một lượt, đồng thời tính metric tổng và theo từng SNR."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_squared_error = 0.0
    total_samples = 0
    per_snr: dict[float, dict[str, float]] = {}

    with torch.no_grad():
        for signals, doa_class, doa_degree, snr_db in tqdm(
            loader,
            desc="Test",
        ):
            signals = signals.to(device, non_blocking=use_pin_memory)
            doa_class = doa_class.to(device, non_blocking=use_pin_memory)
            doa_degree = doa_degree.to(device, non_blocking=use_pin_memory)
            snr_db = snr_db.to(device, non_blocking=use_pin_memory)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                output = model(signals)
                logits = output["logits"]
                loss = criterion(logits, doa_class)

            pred_class = logits.argmax(dim=1)
            pred_degree = (
                cfg.DOA_MIN_DEGREE
                + pred_class.to(dtype=doa_degree.dtype)
                * cfg.DOA_STEP_DEGREE
            )
            squared_error = (pred_degree - doa_degree).square()
            correct = pred_class == doa_class

            batch_size = signals.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += correct.sum().item()
            total_squared_error += squared_error.sum().item()
            total_samples += batch_size

            for snr_value_tensor in torch.unique(snr_db):
                snr_value = float(snr_value_tensor.item())
                mask = snr_db == snr_value_tensor
                stats = per_snr.setdefault(
                    snr_value,
                    {
                        "num_samples": 0,
                        "correct": 0,
                        "squared_error": 0.0,
                    },
                )
                stats["num_samples"] += int(mask.sum().item())
                stats["correct"] += int(correct[mask].sum().item())
                stats["squared_error"] += float(
                    squared_error[mask].sum().item()
                )

    if total_samples == 0:
        raise RuntimeError("Test DataLoader không có sample nào.")

    overall_metrics = {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * total_correct / total_samples,
        "rmse": math.sqrt(total_squared_error / total_samples),
    }

    rows = []
    for snr_value in sorted(per_snr):
        stats = per_snr[snr_value]
        num_samples = int(stats["num_samples"])
        rows.append(
            {
                "snr_db": snr_value,
                "num_samples": num_samples,
                "accuracy": 100.0 * stats["correct"] / num_samples,
                "rmse_degree": math.sqrt(
                    stats["squared_error"] / num_samples
                ),
            }
        )

    return overall_metrics, rows


def write_test_by_snr(path: Path, rows: list[dict[str, float]]) -> None:
    """Ghi metric test theo SNR đã được sort tăng dần."""
    fieldnames = ("snr_db", "num_samples", "accuracy", "rmse_degree")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "snr_db": f"{row['snr_db']:g}",
                    "num_samples": row["num_samples"],
                    "accuracy": f"{row['accuracy']:.6f}",
                    "rmse_degree": f"{row['rmse_degree']:.8f}",
                }
            )


def write_test_summary(
    path: Path,
    parameter_counts: dict[str, int],
    best_epoch: int,
    best_val_rmse: float,
    test_metrics: dict[str, float],
) -> None:
    """Ghi định nghĩa parameter, breakdown và kết quả tổng của test."""
    content = f"""\
MODEL PARAMETER DEFINITION

Một model parameter là một giá trị vô hướng
nằm trong một tensor nn.Parameter.

Buffers không được tính là model parameter.

PARAMETER COUNT

Mamba blocks: {parameter_counts['mamba_blocks']:,}
Pre-LayerNorms: {parameter_counts['pre_layer_norms']:,}
Final LayerNorm: {parameter_counts['final_layer_norm']:,}
Classifier: {parameter_counts['classifier']:,}
Total parameters: {parameter_counts['total']:,}
Trainable parameters: {parameter_counts['trainable']:,}

TRAINING RESULT

Best epoch: {best_epoch}
Best validation RMSE: {best_val_rmse:.8f} degree

TEST RESULT

Test loss: {test_metrics['loss']:.8f}
Test accuracy: {test_metrics['accuracy']:.6f} %
Test RMSE: {test_metrics['rmse']:.8f} degree
"""
    path.write_text(content, encoding="utf-8")


# ============================================================
# 7. MAIN
# ============================================================

def main() -> None:
    """Chạy training, chọn best model bằng validation rồi test một lần."""
    validate_training_configuration()
    device = get_cuda_device()
    set_seed()

    output_paths = get_output_paths()
    prepare_output_directory(output_paths)

    train_loader, val_loader, test_loader, use_pin_memory = (
        create_dataloaders(device)
    )

    model = Mamba1DOAClassifier().to(device)
    parameter_counts = count_model_parameters(model)
    print_parameter_counts(parameter_counts)

    optimizer = create_optimizer(model)
    criterion = create_loss()
    scaler = torch.amp.GradScaler("cuda")

    if cfg.RESUME_TRAINING:
        start_epoch, best_val_rmse, best_epoch = load_resume_checkpoint(
            output_paths["last_checkpoint"],
            output_paths["history"],
            model,
            optimizer,
            scaler,
            device,
        )
        print(f"Resume training từ epoch {start_epoch}.")
    else:
        initialize_history(output_paths["history"])
        start_epoch = 1
        best_val_rmse = math.inf
        best_epoch = 0

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_pin_memory,
            epoch,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_pin_memory,
            description=f"Epoch {epoch}/{cfg.EPOCHS} - Val",
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            is_best = True
        else:
            is_best = False

        learning_rate = optimizer.param_groups[0]["lr"]
        checkpoint = build_checkpoint(
            epoch,
            model,
            optimizer,
            scaler,
            train_metrics,
            val_metrics,
            best_val_rmse,
            best_epoch,
        )

        if is_best:
            torch.save(checkpoint, output_paths["best_model"])

        torch.save(checkpoint, output_paths["last_checkpoint"])
        append_history(
            output_paths["history"],
            epoch,
            train_metrics,
            val_metrics,
            learning_rate,
        )

        print(
            f"Epoch {epoch:02d}/{cfg.EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val Acc: {val_metrics['accuracy']:.2f}% | "
            f"Val RMSE: {val_metrics['rmse']:.4f} deg"
        )

    if not output_paths["best_model"].is_file():
        raise FileNotFoundError(
            f"Không tìm thấy best model sau training: "
            f"{output_paths['best_model']}"
        )

    best_checkpoint = torch.load(
        output_paths["best_model"],
        map_location=device,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics, test_by_snr = evaluate_test(
        model,
        test_loader,
        criterion,
        device,
        use_pin_memory,
    )
    write_test_by_snr(output_paths["test_by_snr"], test_by_snr)
    write_test_summary(
        output_paths["test_summary"],
        parameter_counts,
        int(best_checkpoint["best_epoch"]),
        float(best_checkpoint["best_val_rmse"]),
        test_metrics,
    )

    print("\n===== TEST RESULT =====")
    print(f"Best epoch:   {int(best_checkpoint['best_epoch'])}")
    print(
        "Best val RMSE: "
        f"{float(best_checkpoint['best_val_rmse']):.4f} deg"
    )
    print(f"Test loss:    {test_metrics['loss']:.6f}")
    print(f"Test accuracy:{test_metrics['accuracy']:9.2f}%")
    print(f"Test RMSE:    {test_metrics['rmse']:.4f} deg")
    print("=======================")


if __name__ == "__main__":
    main()
