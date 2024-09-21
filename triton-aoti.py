import torch
torch.set_default_device("cuda")
import os

def triton_transpose_acc(x, y) -> torch.Tensor:
    return (x + y).to(dtype=torch.bfloat16)



torch.manual_seed(0)
N = 64
args = torch.randn(N, N), torch.randn(N, N)
torch._export.aot_compile(triton_transpose_acc, args, {}, options={"aot_inductor.output_path": f"foo.so", "abi_compatible": True})

# Remove rogue dependencies
from subprocess import check_call
check_call("patchelf --remove-needed libtorch.so --remove-needed libtorch_cuda.so --remove-needed libc10_cuda.so --remove-needed libtorch_cpu.so --add-needed libcudart.so foo.so", shell=True)
