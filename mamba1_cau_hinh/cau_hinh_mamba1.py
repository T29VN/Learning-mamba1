"""
Cấu hình dùng chung cho mô hình và quá trình huấn luyện Mamba-1.

File này chỉ định nghĩa các hằng số và đường dẫn. Việc xây dựng mô hình,
đọc dữ liệu và lựa chọn thiết bị sẽ được thực hiện trong các module tương ứng.
"""

from pathlib import Path


# ============================================================
# 1. ĐƯỜNG DẪN PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "tao_du_lieu_gia" / "du_lieu_da_tao"
TRAIN_FILE = DATA_DIR / "train.pt"
VAL_FILE = DATA_DIR / "val.pt"
TEST_FILE = DATA_DIR / "test.pt"

MODEL_SAVE_DIR = PROJECT_ROOT / "mo_hinh_da_huan_luyen" / "mamba1"


# ============================================================
# 2. CẤU HÌNH DỮ LIỆU ĐẦU VÀO
# ============================================================

SEQUENCE_LENGTH = 1024  # số complex sample trong mỗi chuỗi
NUM_IQ = 2              # hai thành phần I và Q
NUM_ANTENNAS = 5        # số anten trong ULA


# ============================================================
# 3. KIẾN TRÚC MAMBA-1
# ============================================================

# Mỗi time step có 2 thành phần I/Q x 5 anten = 10 feature.
D_MODEL = NUM_IQ * NUM_ANTENNAS
D_STATE = 16   # kích thước trạng thái SSM
D_CONV = 4     # kernel size của local convolution
EXPAND = 2     # hệ số mở rộng hidden dimension
NUM_LAYERS = 3  # số Mamba-1 block nối tiếp nhau

POOLING_TYPE = "mean"  # mean pooling theo chiều sequence
DROPOUT = 0.0           # baseline hiện tại không sử dụng dropout
USE_RESIDUAL = True     # residual connection quanh mỗi Mamba block
NORM_POSITION = "pre"   # LayerNorm được áp dụng trước mỗi Mamba block
USE_FINAL_NORM = True   # LayerNorm cuối trước bước pooling


# ============================================================
# 4. BÀI TOÁN DOA
# ============================================================

DOA_MIN_DEGREE = -9
DOA_MAX_DEGREE = 9
DOA_STEP_DEGREE = 1
NUM_CLASSES = 19  # các góc DOA nguyên từ -9 đến 9 độ

_EXPECTED_NUM_CLASSES = (
    (DOA_MAX_DEGREE - DOA_MIN_DEGREE) // DOA_STEP_DEGREE + 1
)
if _EXPECTED_NUM_CLASSES != NUM_CLASSES:
    raise ValueError(
        "NUM_CLASSES không khớp với dải DOA và bước DOA đã cấu hình."
    )


# ============================================================
# 5. HYPERPARAMETERS HUẤN LUYỆN
# ============================================================

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
SEED = 42


# ============================================================
# 6. OPTIMIZER VÀ LOSS
# ============================================================

OPTIMIZER_NAME = "AdamW"
LOSS_NAME = "CrossEntropyLoss"
LR_SCHEDULER = None  # baseline hiện tại không sử dụng scheduler


# ============================================================
# 7. DATALOADER
# ============================================================

NUM_WORKERS = 2
PIN_MEMORY = True

SHUFFLE_TRAIN = True
SHUFFLE_VAL = False
SHUFFLE_TEST = False

DROP_LAST = False


# ============================================================
# 8. TÙY CHỌN RUNTIME/TRAINING
# ============================================================

USE_AMP = True
GRAD_CLIP = None  # baseline hiện tại không gradient clipping
RESUME_TRAINING = False
