import ctypes
import torch
torch.set_default_device("cuda")
import os
from triton import autotune, cdiv, Config, heuristics, jit  # @manual
import triton.language as tl

@autotune(
    configs=[
        Config({"BLOCK_M": 32, "BLOCK_N": 32}),
    ],
    key=["M", "N"],
)


@jit
def _kernel_rms_norm_forward_kernel(
     Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,  # constexpr so the `if` blocks can be optimized out
    BLOCK_SIZE: tl.constexpr,
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # On Llama, only rstd is computed on fp32
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)

    # Gemma computes everything on fp32, and then casts back the output to the original dtype
    if casting_mode == _CASTING_MODE_GEMMA:
        W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(RSTD_ptr, rstd)

    X_row = X_row * rstd

    # On Llama, the multiplication with the weight is done on the original dtype
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)

    Y_row = X_row * (offset + W_row)

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)
@jit
def _kernel_transpose_acc(
    A,
    B,
    C,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_offset = rm[:, None] * N + rn[None, :] * 1
    bt_offset = rm[:, None] * 1 + rn[None, :] * M
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # TODO handle edge case with masks.
    b = tl.load(B + b_offset, mask=mask)
    a = tl.load(A + bt_offset, mask=mask)
    b = b + a
    tl.store(C + bt_offset, b, mask=mask)



def triton_transpose_acc(x, y) -> torch.Tensor:
    assert len(x.shape) == 2
    assert x.shape[0] == y.shape[1]
    assert x.shape[1] == y.shape[0]
    M, N = y.shape
    output = torch.empty(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
    def grid(META):
        return (cdiv(M, META["BLOCK_M"]) * cdiv(N, META["BLOCK_N"]),)
    _kernel_transpose_acc[grid](x, y, output, M, N)
    return output



def test_triton_transpose_acc(args):
    fn_output = triton_transpose_acc(*args)
    lib_fn = torch._export.aot_load(f"./libfoo.so", "cuda")
    lib_output = lib_fn(*args) 
    # TODO test shimmed
    print(fn_output)
    return torch.equal(fn_output, lib_output)



torch.manual_seed(0)
N = 3
args = torch.tensor((0, 2, 4, 6., 8, 10, 12, 14, 16)).reshape(3,3), torch.tensor((0, 2., 4, 6, 8, 10, 12, 14, 16)).reshape(3,3)
torch._export.aot_compile(triton_transpose_acc, args, {}, options={"aot_inductor.output_path": f"libfoo.so", "abi_compatible": True})
assert test_triton_transpose_acc(args)

# Remove rogue dependencies
from subprocess import check_call
check_call("patchelf --remove-needed libtorch.so --remove-needed libtorch_cuda.so --remove-needed libc10_cuda.so --remove-needed libtorch_cpu.so --add-needed libcudart.so libfoo.so", shell=True)
