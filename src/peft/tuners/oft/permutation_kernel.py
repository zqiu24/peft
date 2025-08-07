import torch
import triton
import triton.language as tl
import time
import matplotlib.pyplot as plt
from torch.autograd import Function


@triton.autotune(
    configs=[
        # Define a search space for the autotuner.
        # It will benchmark these configs and pick the fastest one.
        triton.Config({'BLOCK_SIZE_D': 128, 'GROUP_SIZE_R': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 256, 'GROUP_SIZE_R': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 256, 'GROUP_SIZE_R': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE_D': 512, 'GROUP_SIZE_R': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE_D': 512, 'GROUP_SIZE_R': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE_D': 1024, 'GROUP_SIZE_R': 1}, num_warps=16),
    ],
    key=['hidden_dim', 'total_rows'],
)
@triton.jit
def optimized_gather_kernel(
    in_ptr, out_ptr, idx_ptr,
    in_stride_r, in_stride_d,
    out_stride_r, out_stride_d,
    hidden_dim, total_rows,
    BLOCK_SIZE_D: tl.constexpr,
    GROUP_SIZE_R: tl.constexpr,
):
    """
    A generic, optimized kernel to perform a gather operation on the last dimension.
    Equivalent to out[r, c] = in[r, idx[c]].
    This will serve for both the forward and backward permutation passes.
    """
    # 2D grid: one for row groups, one for tiling the hidden dimension
    pid_rg = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Offsets for the current tile of the hidden dimension
    d_offsets = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < hidden_dim

    # Load the indices for the current tile (e.g., perm_indices or inverse_perm_indices)
    # This is done once and reused for all rows in the group.
    source_indices = tl.load(idx_ptr + d_offsets, mask=d_mask)

    # --- Row Grouping ---
    start_row = pid_rg * GROUP_SIZE_R
    r_group_offsets = tl.arange(0, GROUP_SIZE_R)
    current_rows = start_row + r_group_offsets
    r_mask = current_rows < total_rows

    # Prepare pointers for the gather/store operations
    in_rows_ptr = in_ptr + current_rows[:, None] * in_stride_r
    out_rows_ptr = out_ptr + current_rows[:, None] * out_stride_r
    
    in_gather_ptrs = in_rows_ptr + source_indices[None, :] * in_stride_d
    out_store_ptrs = out_rows_ptr + d_offsets[None, :] * out_stride_d

    # Combine masks for safety
    full_mask = r_mask[:, None] & d_mask[None, :]

    # Perform the gather from input and store to output
    vals = tl.load(in_gather_ptrs, mask=full_mask)
    tl.store(out_store_ptrs, vals, mask=full_mask)

class TritonPermute(Function):
    @staticmethod
    def forward(ctx, x, permutation, inv_permutation):
        # The output tensor
        y = torch.empty_like(x)

        # Reshape to 2D for the kernel for simplicity and generality
        x_2d = x.view(-1, x.shape[-1])
        y_2d = y.view(-1, y.shape[-1])
        total_rows, hidden_dim = x_2d.shape

        # Save the forward permutation to compute its inverse in backward
        ctx.save_for_backward(inv_permutation)

        # Define the grid for the kernel launch
        grid = lambda META: (
            triton.cdiv(total_rows, META['GROUP_SIZE_R']),
            triton.cdiv(hidden_dim, META['BLOCK_SIZE_D']),
        )

        # Call the unified kernel for the forward pass
        optimized_gather_kernel[grid](
            x_2d, y_2d, permutation,
            x_2d.stride(0), x_2d.stride(1),
            y_2d.stride(0), y_2d.stride(1),
            hidden_dim, total_rows,
        )
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the forward permutation
        inverse_permutation, = ctx.saved_tensors
        grad_input = None

        if ctx.needs_input_grad[0]:
            # --- Key Step: Compute Inverse Permutation ---
            # The gradient for a permutation is a permutation with the inverse map.

            # Prepare tensors for the kernel
            grad_input = torch.empty_like(grad_output)
            grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
            grad_input_2d = grad_input.view(-1, grad_input.shape[-1])
            total_rows, hidden_dim = grad_output_2d.shape

            grid = lambda META: (
                triton.cdiv(total_rows, META['GROUP_SIZE_R']),
                triton.cdiv(hidden_dim, META['BLOCK_SIZE_D']),
            )
            
            # --- Reuse the SAME kernel for the backward pass ---
            optimized_gather_kernel[grid](
                grad_output_2d, grad_input_2d, inverse_permutation,
                grad_output_2d.stride(0), grad_output_2d.stride(1),
                grad_input_2d.stride(0), grad_input_2d.stride(1),
                hidden_dim, total_rows
            )

        # Return gradients for each input of forward: (x, permutation, inv_permutation)
        return grad_input, None, None

def triton_permute(x, permutation, inv_permutation):
    return TritonPermute.apply(x, permutation, inv_permutation)

def pytorch_permute(x, permutation):
    return x[..., permutation]

def benchmark(shape, device='cuda', repeats=100):
    batch_size, seq_len, hidden_dim = shape
    x = torch.randn(shape, device=device, requires_grad=True)
    permutation = torch.randperm(hidden_dim, device=device)
    inv_permutation = torch.argsort(permutation)
    grad_output = torch.randn(shape, device=device)
    
    # Warmup
    for _ in range(10):
        y = pytorch_permute(x, permutation)
        y.backward(grad_output, retain_graph=True)
        
        y = triton_permute(x, permutation, inv_permutation)
        y.backward(grad_output, retain_graph=True)
    
    # Benchmark PyTorch forward
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        y = pytorch_permute(x, permutation)
    torch.cuda.synchronize()
    pytorch_forward_time = (time.time() - start) / repeats * 1000
    
    # Benchmark PyTorch backward
    y = pytorch_permute(x, permutation)  # Do forward once
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        y.backward(grad_output, retain_graph=True)
        x.grad.zero_()
    torch.cuda.synchronize()
    pytorch_backward_time = (time.time() - start) / repeats * 1000
    
    # Benchmark Triton forward
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        y = triton_permute(x, permutation, inv_permutation)
    torch.cuda.synchronize()
    triton_forward_time = (time.time() - start) / repeats * 1000
    
    # Benchmark Triton backward
    y = triton_permute(x, permutation, inv_permutation)  # Do forward once
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        y.backward(grad_output, retain_graph=True)
        x.grad.zero_()
    torch.cuda.synchronize()
    triton_backward_time = (time.time() - start) / repeats * 1000
    
    # Log comparisons
    print(f"PyTorch - Forward: {pytorch_forward_time:.3f} ms, Backward: {pytorch_backward_time:.3f} ms")
    print(f"Triton  - Forward: {triton_forward_time:.3f} ms, Backward: {triton_backward_time:.3f} ms")
    print(f"Forward speedup: {pytorch_forward_time/triton_forward_time:.2f}x")
    print(f"Backward speedup: {pytorch_backward_time/triton_backward_time:.2f}x")
    print(f"Total speedup: {(pytorch_forward_time + pytorch_backward_time)/(triton_forward_time + triton_backward_time):.2f}x")
    
    return (pytorch_forward_time, pytorch_backward_time), (triton_forward_time, triton_backward_time)
    

def run_benchmarks():
    shapes = [
        (32, 128, 256),
        (64, 256, 512),
        (128, 512, 1024),
        (256, 1024, 2048),
        (1, 2048, 4096),
    ]
    
    results = {'pytorch': [], 'triton': []}
    
    for shape in shapes:
        print(f"\nBenchmarking shape: {shape}")
        benchmark(shape)


if __name__ == '__main__':
    # Verify correctness
    batch_size, seq_len, hidden_dim = 4, 8, 16
    x1 = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', requires_grad=True)
    x2 = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', requires_grad=True)
    x2.data.copy_(x1.data)
    permutation = torch.randperm(hidden_dim, device='cuda')
    inv_permutation = torch.argsort(permutation)
    grad_output = torch.randn_like(x1)
    
    # Forward check
    y_triton = triton_permute(x1, permutation, inv_permutation)
    y_pytorch = pytorch_permute(x2, permutation)
    assert torch.allclose(y_triton, y_pytorch, atol=1e-6), "Forward pass mismatch"
    
    # Backward check
    y_triton.backward(grad_output)
    grad_triton = x1.grad.clone()
    
    y_pytorch.backward(grad_output)
    grad_pytorch = x2.grad.clone()
    
    assert torch.allclose(grad_triton, grad_pytorch, atol=1e-6), "Backward pass mismatch"
    
    print("Correctness verified!")
    run_benchmarks()