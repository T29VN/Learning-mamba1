"""Cấu hình dùng chung cho Elman RNN đối chứng với Mamba-1."""

from pathlib import Path


# ============================================================
# 1. ĐƯỜNG DẪN PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "tao_du_lieu_gia" / "du_lieu_da_tao"
TRAIN_FILE = DATA_DIR / "train.pt"
VAL_FILE = DATA_DIR / "val.pt"
TEST_FILE = DATA_DIR / "test.pt"

MODEL_SAVE_DIR = PROJECT_ROOT / "mo_hinh_da_huan_luyen" / "rnn"


# ============================================================
# 2. CẤU HÌNH DỮ LIỆU ĐẦU VÀO
# ============================================================

SEQUENCE_LENGTH = 1024  # số complex sample trong mỗi chuỗi
NUM_IQ = 2              # hai thành phần I và Q
NUM_ANTENNAS = 5        # số anten trong ULA

# Mỗi time step có 2 thành phần I/Q x 5 anten = 10 feature.
INPUT_SIZE = NUM_IQ * NUM_ANTENNAS


# ============================================================
# 3. KIẾN TRÚC RNN
# ============================================================

HIDDEN_SIZE = 30
NUM_LAYERS = 3

NONLINEARITY = "tanh"
BIDIRECTIONAL = False
BIAS = True

# Ba nn.RNN một layer riêng biệt để chèn pre-LayerNorm giữa các layer.
RNN_LAYER_MODE = "separate_single_layer"
USE_LAYER_NORM = True
NORM_POSITION = "pre"
USE_FINAL_NORM = True

USE_RESIDUAL = False
LEARNABLE_H0 = False
DROPOUT = 0.0
POOLING_TYPE = "mean"


# ============================================================
# 4. BÀI TOÁN DOA
# ============================================================

DOA_MIN_DEGREE = -9
DOA_MAX_DEGREE = 9
DOA_STEP_DEGREE = 1
NUM_CLASSES = 19

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
LR_SCHEDULER = None


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
# 8. RUNTIME/TRAINING
# ============================================================

USE_AMP = True
GRAD_CLIP = None
RESUME_TRAINING = False


# ============================================================
# 9. THAM CHIẾU SỐ PARAMETER
# ============================================================

REFERENCE_MAMBA1_PARAMS = 5509

# Đây là parameter count dự kiến theo kiến trúc đã chốt,
# chưa phải parameter count đã được xác minh bằng model tại runtime.
EXPECTED_RNN_PARAMS = 5769

EXPECTED_PARAMETER_DIFFERENCE = (
    EXPECTED_RNN_PARAMS - REFERENCE_MAMBA1_PARAMS
)
EXPECTED_PARAMETER_DIFFERENCE_PERCENT = (
    EXPECTED_PARAMETER_DIFFERENCE / REFERENCE_MAMBA1_PARAMS * 100.0
)
