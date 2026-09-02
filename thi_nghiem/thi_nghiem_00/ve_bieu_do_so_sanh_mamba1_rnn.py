"""Vẽ biểu đồ so sánh Mamba-1 và RNN từ lần test đầu tiên."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAMBA1_CSV = (
    PROJECT_ROOT
    / "mo_hinh_da_huan_luyen"
    / "mamba1"
    / "lan_train_01"
    / "test_by_snr.csv"
)
RNN_CSV = (
    PROJECT_ROOT
    / "mo_hinh_da_huan_luyen"
    / "rnn"
    / "lan_train_01"
    / "test_by_snr.csv"
)

RMSE_OUTPUT = SCRIPT_DIR / "so_sanh_rmse_theo_snr_lan_test_01.png"
ACCURACY_OUTPUT = SCRIPT_DIR / "so_sanh_accuracy_theo_snr_lan_test_01.png"

REQUIRED_COLUMNS = {
    "snr_db",
    "num_samples",
    "accuracy",
    "rmse_degree",
}


# ============================================================
# 2. ĐỌC VÀ KIỂM TRA DỮ LIỆU
# ============================================================

def load_test_by_snr(path: Path, model_name: str) -> list[dict[str, float]]:
    """Đọc metric test theo SNR và sắp xếp theo SNR tăng dần."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy test_by_snr.csv của {model_name}: {path}"
        )

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        if missing_columns:
            raise ValueError(
                f"File của {model_name} thiếu các cột: {missing_columns}."
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                parsed_row = {
                    "snr_db": float(row["snr_db"]),
                    "num_samples": int(row["num_samples"]),
                    "accuracy": float(row["accuracy"]),
                    "rmse_degree": float(row["rmse_degree"]),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Dữ liệu không hợp lệ tại dòng {line_number} "
                    f"trong file của {model_name}."
                ) from exc

            if parsed_row["num_samples"] <= 0:
                raise ValueError(
                    f"num_samples phải dương tại SNR "
                    f"{parsed_row['snr_db']:g} dB của {model_name}."
                )
            rows.append(parsed_row)

    if not rows:
        raise ValueError(f"File test_by_snr.csv của {model_name} không có dữ liệu.")

    rows.sort(key=lambda item: item["snr_db"])
    snr_values = [row["snr_db"] for row in rows]
    if len(snr_values) != len(set(snr_values)):
        raise ValueError(f"File của {model_name} chứa giá trị SNR bị trùng.")

    return rows


def validate_comparable_snr(
    mamba1_rows: list[dict[str, float]],
    rnn_rows: list[dict[str, float]],
) -> None:
    """Đảm bảo hai mô hình được so sánh trên cùng danh sách SNR."""
    mamba1_snr = [row["snr_db"] for row in mamba1_rows]
    rnn_snr = [row["snr_db"] for row in rnn_rows]

    if mamba1_snr != rnn_snr:
        raise ValueError(
            "Danh sách SNR của Mamba-1 và RNN không khớp; "
            "không thể vẽ biểu đồ so sánh trực tiếp."
        )


def calculate_overall_accuracy(rows: list[dict[str, float]]) -> float:
    """Tính accuracy tổng có trọng số theo số sample tại từng SNR."""
    total_samples = sum(row["num_samples"] for row in rows)
    weighted_correct = sum(
        row["accuracy"] * row["num_samples"] for row in rows
    )
    return weighted_correct / total_samples


def calculate_overall_rmse(rows: list[dict[str, float]]) -> float:
    """Gộp RMSE theo SNR thành RMSE tổng theo toàn bộ sample."""
    total_samples = sum(row["num_samples"] for row in rows)
    total_squared_error = sum(
        row["rmse_degree"] ** 2 * row["num_samples"]
        for row in rows
    )
    return math.sqrt(total_squared_error / total_samples)


# ============================================================
# 3. VẼ BIỂU ĐỒ
# ============================================================

def configure_snr_axis(axis: plt.Axes, snr_values: list[float]) -> None:
    """Thiết lập trục SNR dùng chung cho hai biểu đồ."""
    axis.set_xlabel("SNR (dB)")
    axis.set_xticks(snr_values)
    axis.tick_params(axis="x", rotation=45)
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()


def plot_rmse_comparison(
    mamba1_rows: list[dict[str, float]],
    rnn_rows: list[dict[str, float]],
) -> None:
    """Vẽ RMSE theo SNR của Mamba-1 và RNN."""
    snr_values = [row["snr_db"] for row in mamba1_rows]
    mamba1_rmse = [row["rmse_degree"] for row in mamba1_rows]
    rnn_rmse = [row["rmse_degree"] for row in rnn_rows]

    mamba1_overall = calculate_overall_rmse(mamba1_rows)
    rnn_overall = calculate_overall_rmse(rnn_rows)

    figure, axis = plt.subplots(figsize=(12, 6.5))
    axis.plot(
        snr_values,
        mamba1_rmse,
        color="#1f77b4",
        marker="o",
        linewidth=2,
        markersize=5,
        label=f"Mamba-1 (RMSE tổng: {mamba1_overall:.3f}°)",
    )
    axis.plot(
        snr_values,
        rnn_rmse,
        color="#d62728",
        marker="s",
        linewidth=2,
        markersize=5,
        label=f"RNN (RMSE tổng: {rnn_overall:.3f}°)",
    )

    axis.set_title("So sánh RMSE theo SNR — lần test 01")
    axis.set_ylabel("RMSE (độ)")
    axis.set_ylim(bottom=0)
    configure_snr_axis(axis, snr_values)

    figure.tight_layout()
    figure.savefig(RMSE_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_accuracy_comparison(
    mamba1_rows: list[dict[str, float]],
    rnn_rows: list[dict[str, float]],
) -> None:
    """Vẽ accuracy theo SNR của Mamba-1 và RNN."""
    snr_values = [row["snr_db"] for row in mamba1_rows]
    mamba1_accuracy = [row["accuracy"] for row in mamba1_rows]
    rnn_accuracy = [row["accuracy"] for row in rnn_rows]

    mamba1_overall = calculate_overall_accuracy(mamba1_rows)
    rnn_overall = calculate_overall_accuracy(rnn_rows)

    figure, axis = plt.subplots(figsize=(12, 6.5))
    axis.plot(
        snr_values,
        mamba1_accuracy,
        color="#1f77b4",
        marker="o",
        linewidth=2,
        markersize=5,
        label=f"Mamba-1 (accuracy tổng: {mamba1_overall:.2f}%)",
    )
    axis.plot(
        snr_values,
        rnn_accuracy,
        color="#d62728",
        marker="s",
        linewidth=2,
        markersize=5,
        label=f"RNN (accuracy tổng: {rnn_overall:.2f}%)",
    )

    axis.set_title("So sánh độ chính xác theo SNR — lần test 01")
    axis.set_ylabel("Độ chính xác (%)")
    axis.set_ylim(0, 100)
    configure_snr_axis(axis, snr_values)

    figure.tight_layout()
    figure.savefig(ACCURACY_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(figure)


# ============================================================
# 4. MAIN
# ============================================================

def main() -> None:
    mamba1_rows = load_test_by_snr(MAMBA1_CSV, "Mamba-1")
    rnn_rows = load_test_by_snr(RNN_CSV, "RNN")
    validate_comparable_snr(mamba1_rows, rnn_rows)

    plot_rmse_comparison(mamba1_rows, rnn_rows)
    plot_accuracy_comparison(mamba1_rows, rnn_rows)

    print(f"Đã lưu biểu đồ RMSE: {RMSE_OUTPUT}")
    print(f"Đã lưu biểu đồ accuracy: {ACCURACY_OUTPUT}")


if __name__ == "__main__":
    main()
