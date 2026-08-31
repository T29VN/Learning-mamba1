import torch
from mamba_ssm import Mamba


# ============================================================
# 1. Chọn GPU
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# 2. Các thông số dữ liệu giả giống hình thái MoDANet
# ============================================================
batch_size = 1
sequence_length = 1024
num_iq = 2
num_antennas = 5

# Mỗi thời điểm có:
# 2 thành phần I/Q × 5 anten = 10 giá trị
d_model = num_iq * num_antennas


# ============================================================
# 3. Tạo dữ liệu giả
# ============================================================
# Shape:
# (batch, thời gian, I/Q, anten)
fake_modanet = torch.randn(
    batch_size,
    sequence_length,
    num_iq,
    num_antennas,
    device=device
)
print("\nDữ liệu giả MoDANet:")
print("Shape =", fake_modanet.shape)


# ============================================================
# 4. Đổi sang dạng đầu vào của Mamba
# ============================================================
# (B, 1024, 2, 5)
#          ↓
# (B, 1024, 10)

x = fake_modanet.reshape(
    batch_size,
    sequence_length,
    d_model
).contiguous()

print("\nĐầu vào Mamba-1:")
print("Shape =", x.shape)
#print(x[0,0,:])


# ============================================================
# 5. Tạo một Mamba-1 block
# ============================================================
model = Mamba(
    d_model=10,
    d_state=16,
    d_conv=4,
    expand=2
).to(device)

print("\nMamba-1 block:")
print(model)


# ============================================================
# 6. Cho dữ liệu chạy qua Mamba-1
# ============================================================
model.eval()

with torch.no_grad():
    y = model(x)


# ============================================================
# 7. Kiểm tra đầu ra
# ============================================================
print("\nĐầu ra Mamba-1:")
print("Shape =", y.shape)
print("Device =", y.device)

print("\n10 giá trị đầu tiên của token đầu tiên:")
print(y[0, 0, :])