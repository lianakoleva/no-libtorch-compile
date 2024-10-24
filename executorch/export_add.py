import torch
from torch.export import export
from executorch.exir import to_edge
from executorch.backends.aoti.aoti_partitioner import AotiPartitioner


from triton import autotune, cdiv, Config, heuristics, jit  # @manual
import triton.language as tl

@autotune(
    configs=[
        Config({"BLOCK_M": 32, "BLOCK_N": 32}),
    ],
    key=["M", "N"],
)
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



# Start with a PyTorch model that adds two input tensors (matrices)
class Add(torch.nn.Module):
  def __init__(self):
    super(Add, self).__init__()

  def forward(self, x: torch.Tensor, y: torch.Tensor):
      # return triton_transpose_acc(x, y)
      return x + y

# 1. torch.export: Defines the program with the ATen operator set.
aten_dialect = export(Add(), (torch.ones(10, device="cuda"), torch.ones(10, device="cuda")))
# 2. to_edge: Make optimizations for Edge devices
edge_program = to_edge(aten_dialect)
edge_program = edge_program.to_backend(AotiPartitioner([]))

# 3. to_executorch: Convert the graph to an ExecuTorch program
executorch_program = edge_program.to_executorch()

# 4. Save the compiled .pte program
with open("add.pte", "wb") as file:
    file.write(executorch_program.buffer)

