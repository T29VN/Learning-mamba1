import torch

from einops import rearrange
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from causal_conv1d import causal_conv1d_fn


# ============================================================
# 1. Chọn thiết bị
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# 2. Tạo dữ liệu giả có hình thái giống MoDANet
# ============================================================
batch_size = 1
sequence_length = 1024
num_iq = 2
num_antennas = 5

d_model = num_iq * num_antennas      # 2 x 5 = 10


fake_modanet = torch.randn(
    batch_size,
    sequence_length,
    num_iq,
    num_antennas,
    device=device
)

print("\n[0] Dữ liệu giả MoDANet")
print("    shape =", fake_modanet.shape)
print(fake_modanet)

# ============================================================
# 3. Reshape thành đầu vào Mamba
# ============================================================
hidden_states = fake_modanet.reshape(
    batch_size,
    sequence_length,
    d_model
).contiguous()

print("\n[1] Đầu vào Mamba")
print("    hidden_states =", hidden_states.shape)


# ============================================================
# 4. Tạo Mamba-1 block
#
# use_fast_path=False:
# ép model dùng đường tính toán tường minh thay vì fused kernel
# ============================================================
model = Mamba(
    d_model=10,
    d_state=16,
    d_conv=4,
    expand=2,
    use_fast_path=False
).to(device)

model.eval()

print("\nThông số bên trong Mamba-1:")
print("    d_model =", model.d_model)
print("    d_inner =", model.d_inner)
print("    d_state =", model.d_state)
print("    d_conv  =", model.d_conv)
print("    dt_rank =", model.dt_rank)


# ============================================================
# 5. in_proj
#
# Mã nguồn Mamba thực hiện đồng thời:
# (B, L, D) -> (B, 2*d_inner, L)
# ============================================================
with torch.no_grad():

    xz = rearrange(
        model.in_proj.weight
        @ rearrange(hidden_states, "b l d -> d (b l)"),
        "d (b l) -> b d l",
        l=sequence_length
    )

    if model.in_proj.bias is not None:
        xz = xz + rearrange(
            model.in_proj.bias.to(dtype=xz.dtype),
            "d -> d 1"
        )

    print("\n[2] Sau in_proj")
    print("    xz =", xz.shape)


    # ========================================================
    # 6. Tách thành hai nhánh x và z
    # ========================================================
    x, z = xz.chunk(2, dim=1)

    print("\n[3] Tách xz thành hai nhánh")
    print("    x =", x.shape)
    print("    z =", z.shape)


    # ========================================================
    # 7. Causal Conv1D + SiLU
    # ========================================================
    x = causal_conv1d_fn(
        x=x,
        weight=rearrange(model.conv1d.weight, "d 1 w -> d w"),
        bias=model.conv1d.bias,
        activation=model.activation
    )

    print("\n[4] Sau causal Conv1D + SiLU")
    print("    x =", x.shape)


    # ========================================================
    # 8. Đổi layout để đưa vào x_proj
    #
    # (B, d_inner, L)
    #       ->
    # (B*L, d_inner)
    # ========================================================
    x_for_proj = rearrange(
        x,
        "b d l -> (b l) d"
    )

    print("\n[5] Trước x_proj")
    print("    x_for_proj =", x_for_proj.shape)


    # ========================================================
    # 9. x_proj
    #
    # d_inner
    #    ->
    # dt_rank + d_state + d_state
    # ========================================================
    x_dbl = model.x_proj(x_for_proj)

    print("\n[6] Sau x_proj")
    print("    x_dbl =", x_dbl.shape)


    # ========================================================
    # 10. Tách thành dt, B, C
    # ========================================================
    dt, B, C = torch.split(
        x_dbl,
        [model.dt_rank, model.d_state, model.d_state],
        dim=-1
    )

    print("\n[7] Tách x_dbl thành dt, B, C")
    print("    dt_raw =", dt.shape)
    print("    B_raw  =", B.shape)
    print("    C_raw  =", C.shape)


    # ========================================================
    # 11. dt_proj
    #
    # dt_rank -> d_inner
    #
    # Chú ý:
    # bias chưa cộng ở đây.
    # Nó sẽ được đưa vào selective_scan sau.
    # ========================================================
    dt = model.dt_proj.weight @ dt.t()

    dt = rearrange(
        dt,
        "d (b l) -> b d l",
        l=sequence_length
    )

    print("\n[8] Sau dt_proj")
    print("    dt =", dt.shape)


    # ========================================================
    # 12. Đưa B và C về layout mà Selective Scan cần
    # ========================================================
    B = rearrange(
        B,
        "(b l) dstate -> b dstate l",
        l=sequence_length
    ).contiguous()

    C = rearrange(
        C,
        "(b l) dstate -> b dstate l",
        l=sequence_length
    ).contiguous()

    print("\n[9] B và C trước Selective Scan")
    print("    B =", B.shape)
    print("    C =", C.shape)


    # ========================================================
    # 13. Ma trận A của SSM
    # ========================================================
    A = -torch.exp(model.A_log.float())

    print("\n[10] Ma trận A của SSM")
    print("    A =", A.shape)


    # ========================================================
    # 14. Selective Scan
    #
    # Đây là lõi Selective SSM — SSM chọn lọc.
    # ========================================================
    y = selective_scan_fn(
        x,
        dt,
        A,
        B,
        C,
        model.D.float(),
        z=z,
        delta_bias=model.dt_proj.bias.float(),
        delta_softplus=True,
        return_last_state=False
    )

    print("\n[11] Sau Selective Scan")
    print("    y =", y.shape)


    # ========================================================
    # 15. Đưa layout về (B, L, d_inner)
    # ========================================================
    y = rearrange(
        y,
        "b d l -> b l d"
    )

    print("\n[12] Sau transpose / rearrange")
    print("    y =", y.shape)


    # ========================================================
    # 16. out_proj
    #
    # d_inner -> d_model
    # ========================================================
    output = model.out_proj(y)

    print("\n[13] Sau out_proj")
    print("    output =", output.shape)


print("\nHoàn thành.")