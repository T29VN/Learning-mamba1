#!/usr/bin/env python3
"""
tao_du_lieu_gia_modanet.py
==========================
Tạo bộ dữ liệu tín hiệu vô tuyến giả theo mô hình MoDANet thu gọn
cho bài toán DOA classification.

Ý tưởng lấy từ mô hình tín hiệu MoDANet (ULA, LOS + NLOS multipath),
nhưng các tham số (DOA range, SNR range, Fs, modulation, …) được tùy chỉnh
riêng cho thử nghiệm Mamba-1 hiện tại.

Lưu ý quan trọng:
- Fs = 100 MHz là tham số do thử nghiệm này quy định,
  KHÔNG phải giá trị được công bố trong bài MoDANet gốc.
- DOA range [-9°, 9°], SNR range [-20 dB, 5 dB] cũng là tham số tùy chỉnh.
"""

import os
import sys
import json
import csv
import random
import datetime
from pathlib import Path

import numpy as np
import torch

# ============================================================
# 1. CẤU HÌNH
# ============================================================

SEED = 42

# --- Antenna array ---
M = 5  # số anten trong ULA
# d = lambda/2  →  steering vector: a_m(θ) = exp(-j π m sin(θ))

# --- DOA ---
DOA_MIN = -9    # degree
DOA_MAX = 9     # degree
DOA_STEP = 1    # degree
DOA_LIST = list(range(DOA_MIN, DOA_MAX + 1, DOA_STEP))
NUM_DOA_CLASSES = len(DOA_LIST)  # 19

# --- SNR ---
SNR_MIN = -20   # dB
SNR_MAX = 5     # dB
SNR_STEP = 1    # dB
SNR_LIST = list(range(SNR_MIN, SNR_MAX + 1, SNR_STEP))
NUM_SNR = len(SNR_LIST)  # 26

# --- Signal ---
N = 1024  # complex samples per frame
MAX_DELAY_SAMPLES = 300
SOURCE_LENGTH = N + MAX_DELAY_SAMPLES  # 1324

# --- Sampling rate ---
FS = 100e6  # 100 MHz
TS_NS = 10  # 10 ns  (= 1e9 / FS)

# --- NLOS ---
NLOS_MIN = 1
NLOS_MAX = 10
NLOS_ANGLE_MIN = -60.0   # degree
NLOS_ANGLE_MAX = 60.0    # degree
NLOS_ATTEN_MIN = -50.0   # dB
NLOS_ATTEN_MAX = -1.0    # dB
NLOS_DELAY_NS_MIN = 1.0  # ns
NLOS_DELAY_NS_MAX = 3000.0  # ns

# --- Samples ---
SAMPLES_PER_PAIR = 20
TRAIN_PER_PAIR = 16
VAL_PER_PAIR = 2
TEST_PER_PAIR = 2

# --- Output paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "du_lieu_da_tao"
NHAN_DIR = OUTPUT_DIR / "nhan"
METADATA_DIR = OUTPUT_DIR / "metadata"


# ============================================================
# 2. HÀM TIỆN ÍCH
# ============================================================

def set_seed(seed: int):
    """Đặt seed cho tất cả nguồn random để đảm bảo reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def steering_vector(theta_deg: float) -> np.ndarray:
    """
    Tính steering vector cho ULA với d = λ/2.

    a_m(θ) = exp(-j π m sin(θ)),  m = 0, 1, ..., M-1

    Parameters
    ----------
    theta_deg : float
        Góc DOA tính bằng degree.

    Returns
    -------
    np.ndarray
        Steering vector, shape (M,), dtype complex128.
    """
    theta_rad = np.deg2rad(theta_deg)
    m = np.arange(M)  # [0, 1, 2, 3, 4]
    return np.exp(-1j * np.pi * m * np.sin(theta_rad))


def generate_qpsk_source(rng: np.random.Generator, length: int) -> np.ndarray:
    """
    Sinh chuỗi QPSK symbols ngẫu nhiên.

    Mỗi symbol thuộc { (±1 ± j) / sqrt(2) }.
    Công suất trung bình: E[|s|²] = 1.

    Parameters
    ----------
    rng : np.random.Generator
        Nguồn random chính.
    length : int
        Chiều dài chuỗi cần sinh.

    Returns
    -------
    np.ndarray
        QPSK source, shape (length,), dtype complex128.
    """
    # 4 constellation points
    constellation = np.array([
        (1 + 1j) / np.sqrt(2),
        (1 - 1j) / np.sqrt(2),
        (-1 + 1j) / np.sqrt(2),
        (-1 - 1j) / np.sqrt(2),
    ])
    indices = rng.integers(0, 4, size=length)
    return constellation[indices]


def db_to_amplitude(db: float) -> float:
    """
    Chuyển dB sang amplitude ratio.

    alpha = 10^(dB / 20)

    KHÔNG dùng 10^(dB/10) — đó là cho power ratio.
    """
    return 10.0 ** (db / 20.0)


def delay_ns_to_samples(delay_ns: float) -> int:
    """
    Chuyển delay vật lý (ns) sang số sample nguyên.

    delay_samples = max(1, round(delay_ns / Ts_ns))

    Ts = 10 ns  (Fs = 100 MHz).
    """
    delay_samples = round(delay_ns / TS_NS)
    return max(1, delay_samples)


def generate_one_sample(
    doa_degree: int,
    snr_db: int,
    rng: np.random.Generator,
) -> tuple:
    """
    Sinh một sample tín hiệu theo mô hình LOS + NLOS + AWGN.

    Parameters
    ----------
    doa_degree : int
        Góc DOA chính (LOS), đơn vị degree.
    snr_db : int
        Tỉ số SNR, đơn vị dB.
    rng : np.random.Generator
        Nguồn random chính.

    Returns
    -------
    signal_iq : np.ndarray
        Tín hiệu I/Q, shape (1024, 2, 5), dtype float32.
    nlos_metadata : dict
        Metadata chứa thông tin NLOS cho sample này.
    """
    # ----- 1. Sinh source_long QPSK -----
    source_long = generate_qpsk_source(rng, SOURCE_LENGTH)

    # ----- 2. Cắt LOS frame -----
    s_los = source_long[MAX_DELAY_SAMPLES: MAX_DELAY_SAMPLES + N]

    # ----- 3. Steering vector LOS -----
    a_los = steering_vector(doa_degree)

    # ----- 4. Tín hiệu LOS tại array -----
    # X_LOS[n, m] = s_los[n] * a_los[m]
    X_LOS = s_los[:, np.newaxis] * a_los[np.newaxis, :]  # (1024, 5)

    # ----- 5. Khởi tạo X_clean -----
    X_clean = X_LOS.copy()

    # ----- 6. Sinh và cộng NLOS -----
    num_nlos = int(rng.integers(NLOS_MIN, NLOS_MAX + 1))

    nlos_list = []
    for _ in range(num_nlos):
        # Góc NLOS: Uniform(-60, 60) degree — continuous
        angle_deg = float(rng.uniform(NLOS_ANGLE_MIN, NLOS_ANGLE_MAX))

        # Attenuation: Uniform(-50, -1) dB
        atten_db = float(rng.uniform(NLOS_ATTEN_MIN, NLOS_ATTEN_MAX))
        atten_linear = db_to_amplitude(atten_db)

        # Delay: Uniform(1, 3000) ns
        delay_ns = float(rng.uniform(NLOS_DELAY_NS_MIN, NLOS_DELAY_NS_MAX))
        delay_samp = delay_ns_to_samples(delay_ns)

        # Lấy delayed source từ source_long (dùng lịch sử, không zero pad)
        start_idx = MAX_DELAY_SAMPLES - delay_samp
        s_nlos = source_long[start_idx: start_idx + N]

        # Steering vector NLOS
        a_nlos = steering_vector(angle_deg)

        # NLOS component: alpha * s_delayed * a(theta)
        X_NLOS_p = atten_linear * (s_nlos[:, np.newaxis] * a_nlos[np.newaxis, :])

        # Cộng vào X_clean
        X_clean += X_NLOS_p

        # Lưu metadata
        nlos_list.append({
            "angle_degree": round(angle_deg, 4),
            "attenuation_db": round(atten_db, 4),
            "attenuation_linear": round(atten_linear, 5),
            "delay_ns": round(delay_ns, 4),
            "delay_samples": delay_samp,
        })

    # ----- 7. Tính signal power (LOS + all NLOS) -----
    signal_power = np.mean(np.abs(X_clean) ** 2)

    # ----- 8. Sinh AWGN theo SNR -----
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise = (
        np.sqrt(noise_power / 2.0)
        * (rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M)))
    )

    # ----- 9. Cộng AWGN -----
    X_received = X_clean + noise
    # KHÔNG normalize lại sau khi cộng AWGN.

    # ----- 10. Tách I/Q -----
    I = np.real(X_received)  # (1024, 5)
    Q = np.imag(X_received)  # (1024, 5)

    signal_iq = np.stack([I, Q], axis=1)  # (1024, 2, 5)

    # ----- 11. Convert float32 -----
    signal_iq = signal_iq.astype(np.float32)

    # ----- Metadata -----
    nlos_metadata = {
        "num_nlos": num_nlos,
        "nlos": nlos_list,
    }

    return signal_iq, nlos_metadata


# ============================================================
# 3. CHỐNG GHI ĐÈ
# ============================================================

def check_existing_files() -> list:
    """
    Kiểm tra xem các file dataset chính đã tồn tại chưa.

    Returns
    -------
    list
        Danh sách các file đã tồn tại. Rỗng nếu không có.
    """
    critical_files = [
        OUTPUT_DIR / "train.pt",
        OUTPUT_DIR / "val.pt",
        OUTPUT_DIR / "test.pt",
        NHAN_DIR / "train_labels.csv",
        NHAN_DIR / "val_labels.csv",
        NHAN_DIR / "test_labels.csv",
    ]
    existing = [str(f) for f in critical_files if f.exists()]
    return existing


# ============================================================
# 4. LƯU DỮ LIỆU
# ============================================================

def save_pt_dataset(
    split_name: str,
    signals: np.ndarray,
    doa_classes: list,
    doa_degrees: list,
    snr_dbs: list,
):
    """
    Lưu một split thành file .pt chứa dictionary.

    Parameters
    ----------
    split_name : str
        Tên split: "train", "val", hoặc "test".
    signals : np.ndarray
        Mảng tín hiệu I/Q, shape (num_samples, 1024, 2, 5).
    doa_classes : list
        Danh sách DOA class (int).
    doa_degrees : list
        Danh sách DOA degree (int/float).
    snr_dbs : list
        Danh sách SNR dB (int/float).
    """
    data = {
        "signals": torch.tensor(signals, dtype=torch.float32),
        "doa_class": torch.tensor(doa_classes, dtype=torch.long),
        "doa_degree": torch.tensor(doa_degrees, dtype=torch.float32),
        "snr_db": torch.tensor(snr_dbs, dtype=torch.float32),
    }
    filepath = OUTPUT_DIR / f"{split_name}.pt"
    torch.save(data, filepath)
    print(f"  Saved: {filepath}")


def save_labels_csv(
    split_name: str,
    doa_classes: list,
    doa_degrees: list,
    snr_dbs: list,
    num_nlos_list: list,
):
    """
    Lưu nhãn thành file CSV.

    Columns: sample_id, doa_class, doa_degree, snr_db, num_nlos
    """
    filepath = NHAN_DIR / f"{split_name}_labels.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "doa_class", "doa_degree", "snr_db", "num_nlos"])
        for i in range(len(doa_classes)):
            writer.writerow([i, doa_classes[i], doa_degrees[i], snr_dbs[i], num_nlos_list[i]])
    print(f"  Saved: {filepath}")


def save_metadata_jsonl(
    split_name: str,
    doa_classes: list,
    doa_degrees: list,
    snr_dbs: list,
    all_nlos_metadata: list,
):
    """
    Lưu metadata NLOS thành file JSONL (1 dòng JSON / sample).
    """
    filepath = METADATA_DIR / f"{split_name}_nlos.jsonl"
    with open(filepath, "w") as f:
        for i in range(len(doa_classes)):
            record = {
                "sample_id": i,
                "doa_degree": doa_degrees[i],
                "doa_class": doa_classes[i],
                "snr_db": snr_dbs[i],
                "num_nlos": all_nlos_metadata[i]["num_nlos"],
                "nlos": all_nlos_metadata[i]["nlos"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved: {filepath}")


def write_dataset_info():
    """
    Tạo file thong_tin_dataset.txt mô tả dataset.
    """
    filepath = OUTPUT_DIR / "thong_tin_dataset.txt"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""\
Dataset: Synthetic MoDANet-style DOA dataset

Seed: {SEED}

Created: {timestamp}

Task:
DOA classification

Modulation:
QPSK only

DOA:
min = {DOA_MIN} degree
max = {DOA_MAX} degree
step = {DOA_STEP} degree
num_classes = {NUM_DOA_CLASSES}

SNR:
min = {SNR_MIN} dB
max = {SNR_MAX} dB
step = {SNR_STEP} dB
num_snr = {NUM_SNR}

Array:
ULA
num_antennas = {M}
spacing = lambda/2

Signal:
length = {N} complex samples
saved_shape = {N} x 2 x {M}
saved_dtype = float32

Sampling:
Fs = 100 MHz
Ts = 10 ns

NLOS:
number = Uniform integer [{NLOS_MIN},{NLOS_MAX}]
angle = Uniform [{NLOS_ANGLE_MIN},{NLOS_ANGLE_MAX}] degree
attenuation = Uniform [{NLOS_ATTEN_MIN},{NLOS_ATTEN_MAX}] dB
delay = Uniform [{NLOS_DELAY_NS_MIN},{NLOS_DELAY_NS_MAX}] ns
delay_method = integer sample rounding

Samples per DOA-SNR pair:
{SAMPLES_PER_PAIR}

Split per pair:
train = {TRAIN_PER_PAIR}
validation = {VAL_PER_PAIR}
test = {TEST_PER_PAIR}

Total:
train = {NUM_DOA_CLASSES * NUM_SNR * TRAIN_PER_PAIR}
validation = {NUM_DOA_CLASSES * NUM_SNR * VAL_PER_PAIR}
test = {NUM_DOA_CLASSES * NUM_SNR * TEST_PER_PAIR}
all = {NUM_DOA_CLASSES * NUM_SNR * SAMPLES_PER_PAIR}

Important:
Fs = 100 MHz is a custom simulation parameter,
not a value claimed to come from the MoDANet paper.
"""
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  Saved: {filepath}")


# ============================================================
# 5. VERIFICATION
# ============================================================

def verify_dataset():
    """
    Kiểm tra tính hợp lệ của dataset sau khi tạo.

    Trả về True nếu tất cả kiểm tra đều passed, False nếu có lỗi.
    """
    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)

    all_passed = True

    expected = {
        "train": NUM_DOA_CLASSES * NUM_SNR * TRAIN_PER_PAIR,   # 7904
        "val": NUM_DOA_CLASSES * NUM_SNR * VAL_PER_PAIR,       # 988
        "test": NUM_DOA_CLASSES * NUM_SNR * TEST_PER_PAIR,     # 988
    }

    for split_name, expected_count in expected.items():
        print(f"\n--- {split_name} ---")

        # Load .pt
        pt_path = OUTPUT_DIR / f"{split_name}.pt"
        data = torch.load(pt_path, weights_only=False)

        signals = data["signals"]
        doa_class = data["doa_class"]
        doa_degree = data["doa_degree"]
        snr_db = data["snr_db"]

        # 36.1 Số lượng
        if signals.shape[0] != expected_count:
            print(f"  FAIL: count = {signals.shape[0]}, expected {expected_count}")
            all_passed = False
        else:
            print(f"  OK: count = {expected_count}")

        # 36.2 Shape
        expected_shape = (expected_count, N, 2, M)
        if tuple(signals.shape) != expected_shape:
            print(f"  FAIL: shape = {tuple(signals.shape)}, expected {expected_shape}")
            all_passed = False
        else:
            print(f"  OK: shape = {expected_shape}")

        # 36.3 NaN
        if torch.isnan(signals).any():
            print("  FAIL: signals contain NaN")
            all_passed = False
        else:
            print("  OK: no NaN")

        # 36.4 Inf
        if torch.isinf(signals).any():
            print("  FAIL: signals contain Inf")
            all_passed = False
        else:
            print("  OK: no Inf")

        # 36.5 DOA class range & completeness
        unique_classes = set(doa_class.tolist())
        expected_classes = set(range(NUM_DOA_CLASSES))
        if unique_classes != expected_classes:
            print(f"  FAIL: DOA classes = {sorted(unique_classes)}, expected 0..{NUM_DOA_CLASSES-1}")
            all_passed = False
        else:
            print(f"  OK: all {NUM_DOA_CLASSES} DOA classes present")

        # 36.6 DOA degree range
        unique_degrees = set(doa_degree.tolist())
        expected_degrees = set(float(d) for d in DOA_LIST)
        if unique_degrees != expected_degrees:
            print(f"  FAIL: DOA degrees mismatch")
            all_passed = False
        else:
            print(f"  OK: DOA degrees = [{DOA_MIN}..{DOA_MAX}]")

        # 36.7 SNR range & completeness
        unique_snrs = set(snr_db.tolist())
        expected_snrs = set(float(s) for s in SNR_LIST)
        if unique_snrs != expected_snrs:
            print(f"  FAIL: SNR values mismatch")
            all_passed = False
        else:
            print(f"  OK: all {NUM_SNR} SNR levels present")

        # 36.8 Cân bằng theo từng cặp (DOA, SNR)
        per_pair_key = "train" if split_name == "train" else split_name
        per_pair_expected = {
            "train": TRAIN_PER_PAIR,
            "val": VAL_PER_PAIR,
            "test": TEST_PER_PAIR,
        }[split_name]

        pair_counts = {}
        for i in range(len(doa_class)):
            key = (int(doa_class[i].item()), float(snr_db[i].item()))
            pair_counts[key] = pair_counts.get(key, 0) + 1

        balance_ok = True
        for key, count in pair_counts.items():
            if count != per_pair_expected:
                print(f"  FAIL: pair {key} has {count} samples, expected {per_pair_expected}")
                balance_ok = False
                all_passed = False

        if balance_ok:
            print(f"  OK: all (DOA, SNR) pairs have exactly {per_pair_expected} samples")

        # 36.9 CSV
        csv_path = NHAN_DIR / f"{split_name}_labels.csv"
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            csv_rows = list(reader)
        csv_data_rows = len(csv_rows) - 1  # subtract header
        if csv_data_rows != expected_count:
            print(f"  FAIL: CSV rows = {csv_data_rows}, expected {expected_count}")
            all_passed = False
        else:
            print(f"  OK: CSV rows = {expected_count}")

        # 36.10 Metadata JSONL
        jsonl_path = METADATA_DIR / f"{split_name}_nlos.jsonl"
        with open(jsonl_path, "r") as f:
            jsonl_lines = f.readlines()

        if len(jsonl_lines) != expected_count:
            print(f"  FAIL: JSONL records = {len(jsonl_lines)}, expected {expected_count}")
            all_passed = False
        else:
            print(f"  OK: JSONL records = {expected_count}")

        # Check num_nlos validity in JSONL
        jsonl_ok = True
        for line_idx, line in enumerate(jsonl_lines):
            record = json.loads(line)
            nn = record.get("num_nlos", 0)
            nlos_arr = record.get("nlos", [])
            if nn < 1 or nn > 10:
                print(f"  FAIL: JSONL record {line_idx} has num_nlos = {nn}")
                jsonl_ok = False
                all_passed = False
                break
            if len(nlos_arr) != nn:
                print(f"  FAIL: JSONL record {line_idx}: len(nlos)={len(nlos_arr)} != num_nlos={nn}")
                jsonl_ok = False
                all_passed = False
                break

        if jsonl_ok:
            print(f"  OK: JSONL metadata valid (num_nlos in [1,10], len(nlos) matches)")

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 50)

    return all_passed


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("=" * 50)
    print("Synthetic MoDANet-style DOA Dataset Generator")
    print("=" * 50)

    # --- Chống ghi đè ---
    existing = check_existing_files()
    if existing:
        print("\nERROR: Các file sau đã tồn tại:")
        for f in existing:
            print(f"  - {f}")
        print("\nDừng chương trình để bảo vệ dataset cũ.")
        print("Nếu muốn tạo lại, hãy xóa thủ công các file trên.")
        sys.exit(1)

    # --- Seed ---
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    print(f"\nSeed: {SEED}")

    # --- Tạo thư mục ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NHAN_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # --- Containers cho 3 splits ---
    split_data = {
        "train": {
            "signals": [],
            "doa_classes": [],
            "doa_degrees": [],
            "snr_dbs": [],
            "num_nlos_list": [],
            "nlos_metadata": [],
        },
        "val": {
            "signals": [],
            "doa_classes": [],
            "doa_degrees": [],
            "snr_dbs": [],
            "num_nlos_list": [],
            "nlos_metadata": [],
        },
        "test": {
            "signals": [],
            "doa_classes": [],
            "doa_degrees": [],
            "snr_dbs": [],
            "num_nlos_list": [],
            "nlos_metadata": [],
        },
    }

    # --- Sinh dataset ---
    total_conditions = NUM_DOA_CLASSES * NUM_SNR  # 494
    condition_idx = 0

    print(f"\nBắt đầu sinh {NUM_DOA_CLASSES * NUM_SNR * SAMPLES_PER_PAIR} samples...")
    print(f"  DOA: {DOA_MIN}° → {DOA_MAX}° ({NUM_DOA_CLASSES} classes)")
    print(f"  SNR: {SNR_MIN} dB → {SNR_MAX} dB ({NUM_SNR} levels)")
    print(f"  Samples per pair: {SAMPLES_PER_PAIR}")
    print(f"  Split: train={TRAIN_PER_PAIR}, val={VAL_PER_PAIR}, test={TEST_PER_PAIR}")
    print()

    for doa in DOA_LIST:
        doa_class = int(doa - DOA_MIN)  # doa + 9

        for snr in SNR_LIST:
            condition_idx += 1
            print(f"  [{condition_idx}/{total_conditions}] DOA={doa:+d} deg | SNR={snr:+d} dB")

            # Sinh 20 samples cho cặp (DOA, SNR) này
            samples_for_pair = []
            for _ in range(SAMPLES_PER_PAIR):
                signal_iq, nlos_meta = generate_one_sample(doa, snr, rng)
                samples_for_pair.append((signal_iq, nlos_meta))

            # Chia: 16 train, 2 val, 2 test
            # Thứ tự đã cố định vì rng deterministic
            train_samples = samples_for_pair[:TRAIN_PER_PAIR]
            val_samples = samples_for_pair[TRAIN_PER_PAIR:TRAIN_PER_PAIR + VAL_PER_PAIR]
            test_samples = samples_for_pair[TRAIN_PER_PAIR + VAL_PER_PAIR:]

            for split_name, split_samples in [
                ("train", train_samples),
                ("val", val_samples),
                ("test", test_samples),
            ]:
                for sig_iq, nlos_meta in split_samples:
                    split_data[split_name]["signals"].append(sig_iq)
                    split_data[split_name]["doa_classes"].append(doa_class)
                    split_data[split_name]["doa_degrees"].append(doa)
                    split_data[split_name]["snr_dbs"].append(snr)
                    split_data[split_name]["num_nlos_list"].append(nlos_meta["num_nlos"])
                    split_data[split_name]["nlos_metadata"].append(nlos_meta)

    # --- Lưu dữ liệu ---
    print("\n" + "=" * 50)
    print("SAVING FILES")
    print("=" * 50)

    for split_name in ["train", "val", "test"]:
        sd = split_data[split_name]
        signals_np = np.array(sd["signals"], dtype=np.float32)

        print(f"\n{split_name}:")
        save_pt_dataset(
            split_name,
            signals_np,
            sd["doa_classes"],
            sd["doa_degrees"],
            sd["snr_dbs"],
        )
        save_labels_csv(
            split_name,
            sd["doa_classes"],
            sd["doa_degrees"],
            sd["snr_dbs"],
            sd["num_nlos_list"],
        )
        save_metadata_jsonl(
            split_name,
            sd["doa_classes"],
            sd["doa_degrees"],
            sd["snr_dbs"],
            sd["nlos_metadata"],
        )

    print()
    write_dataset_info()

    # --- Summary ---
    print("\n" + "=" * 50)
    print("DATASET GENERATION COMPLETED")
    print("=" * 50)

    for split_name in ["train", "val", "test"]:
        sd = split_data[split_name]
        n_samples = len(sd["doa_classes"])
        print(f"\n{split_name.capitalize()}:")
        print(f"  {n_samples} samples")
        print(f"  shape = [{n_samples}, {N}, 2, {M}]")

    total = sum(len(split_data[s]["doa_classes"]) for s in ["train", "val", "test"])
    print(f"\nTotal: {total} samples")
    print(f"\nFiles saved to: {OUTPUT_DIR}")

    # --- Verification ---
    passed = verify_dataset()

    if not passed:
        print("\nWARNING: Một số kiểm tra đã thất bại!")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
