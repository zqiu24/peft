import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import triton
import triton.language as tl
from torch.autograd import Function

@triton.autotune(
    configs=[
        # Test smaller stages/warps for smaller blocks
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=4, num_warps=2),

        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=16),

        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=16),
    ],
    key=['N'], # Autotune based on matrix size N
)
@triton.jit
def skew_symmetric_forward_kernel_optimized(
    vec_ptr,
    mat_ptr,
    N,
    stride_vec_batch,
    stride_vec_element,
    stride_mat_batch,
    stride_mat_row,
    stride_mat_col,
    BLOCK_SIZE: tl.constexpr,
): 
    # 3D program IDs: batch, row block, column block
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Offset calculations for matrix blocks
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid matrix indices
    mask_m = offs_m < N
    mask_n = offs_n < N
    full_mask = mask_m[:, None] & mask_n[None, :]

    # Create 2D indices [BLOCK_SIZE, BLOCK_SIZE]
    i = offs_m[:, None]  # [BLOCK_SIZE, 1]
    j = offs_n[None, :]  # [1, BLOCK_SIZE]
    
    # Upper triangle processing
    upper_mask = (i < j) & full_mask
    
    # Vector index calculation for upper triangle
    upper_idx = i * (2 * N - i - 1) // 2 + (j - i - 1)

    # Batch-aware pointer arithmetic
    vec_batch_ptr = vec_ptr + pid_batch * stride_vec_batch
    vec_ptrs = vec_batch_ptr + upper_idx * stride_vec_element
    
    # Load upper triangle values
    upper_vals = tl.load(vec_ptrs, mask=upper_mask, other=0.0)
    
    # Matrix pointer calculations for current batch
    mat_batch_ptr = mat_ptr + pid_batch * stride_mat_batch
    mat_ptrs_upper = mat_batch_ptr + i * stride_mat_row + j * stride_mat_col
    mat_ptrs_lower = mat_batch_ptr + j * stride_mat_row + i * stride_mat_col
    
    # Store upper values and their negatives (skew-symmetric)
    tl.store(mat_ptrs_upper, upper_vals, mask=upper_mask)
    tl.store(mat_ptrs_lower, -upper_vals, mask=upper_mask)
    
    # Zero out diagonal elements
    diag_mask = (i == j) & full_mask
    diag_ptrs = mat_batch_ptr + i * stride_mat_row + j * stride_mat_col
    tl.store(diag_ptrs, tl.zeros((BLOCK_SIZE, BLOCK_SIZE), 
                               dtype=vec_ptr.dtype.element_ty), 
             mask=diag_mask)


@triton.autotune(
    configs=[
        # Test smaller stages/warps for smaller blocks
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=4, num_warps=2),

        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=4),

        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=16),

        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=16),
    ],
    key=['N'], # Autotune based on matrix size N
)
@triton.jit
def skew_symmetric_backward_kernel_optimized(
    grad_mat_ptr,
    grad_vec_ptr,
    N: tl.int32,
    F: tl.int32,
    stride_mat_batch,
    stride_mat_row,
    stride_mat_col,
    stride_vec_batch,
    stride_vec_element,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: batch index x element blocks
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)

    # Element offsets within current batch
    offs = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    k = offs
    mask = k < F  # Filter elements within vector length
    
    # Convert to float for calculations
    N_float = N.to(tl.float32)
    k_float = k.to(tl.float32)
    
    # Quadratic formula to find matrix indices (i, j) from vector index k
    a = 2.0 * N_float - 1.0
    sqrt_val = tl.sqrt(a * a - 8.0 * k_float)
    i_float = (a - sqrt_val) / 2.0
    i = tl.floor(i_float).to(tl.int32)
    
    # Calculate column index j
    triangular_num = i * (2 * N - i - 1) // 2
    j = k - triangular_num + i + 1
    
    # Validate indices
    valid = (i >= 0) & (j < N) & (i < j)
    mask = mask & valid
    
    # Matrix pointer calculations for current batch
    mat_batch_ptr = grad_mat_ptr + pid_batch * stride_mat_batch
    upper_ptr = mat_batch_ptr + i * stride_mat_row + j * stride_mat_col
    lower_ptr = mat_batch_ptr + j * stride_row + i * stride_col
    
    # Load gradients from upper and lower triangle
    grad_upper = tl.load(upper_ptr, mask=mask, other=0.0)
    grad_lower = tl.load(lower_ptr, mask=mask, other=0.0)
    
    # Compute vector gradient (upper - lower due to skew-symmetry)
    grad_vec_val = grad_upper - grad_lower
    
    # Vector pointer calculations
    vec_batch_ptr = grad_vec_ptr + pid_batch * stride_vec_batch
    vec_ptr = vec_batch_ptr + k * stride_vec_element
    
    # Store results
    tl.store(vec_ptr, grad_vec_val, mask=mask)

# --------------------------
# Autograd Function
# --------------------------
class SkewSymmetric(Function):
    @staticmethod
    def forward(ctx, vec, N):
        # Calculate matrix size from vector length
        vec_size = vec.shape[1]
        batch_size = vec.shape[0]
        mat = torch.empty((batch_size, N, N), 
                            device=vec.device, dtype=vec.dtype)

        # Configure kernel launch parameters
        grid = lambda meta: (
            batch_size,
            triton.cdiv(N, meta['BLOCK_SIZE']),
            triton.cdiv(N, meta['BLOCK_SIZE'])
        )

        skew_symmetric_forward_kernel_optimized[grid](
            vec_ptr=vec,
            mat_ptr=mat,
            N=N,
            stride_vec_batch=vec.stride(0),
            stride_vec_element=vec.stride(1),
            stride_mat_batch=mat.stride(0),
            stride_mat_row=mat.stride(1),
            stride_mat_col=mat.stride(2),
            # BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(vec)
        ctx.N = N
        return mat

    @staticmethod
    def backward(ctx, grad_output):
        vec, = ctx.saved_tensors
        N = ctx.N
        batch_size, F = vec.shape
        grad_vec = torch.zeros_like(vec)
        
        # Configure kernel launch parameters
        F = N * (N - 1) // 2
        total_global_elements = batch_size * F
        grid = lambda meta: (triton.cdiv(total_global_elements, meta['BLOCK_SIZE']), )
        # grid = lambda meta: (batch_size, triton.cdiv(F, meta['BLOCK_SIZE']))
        
        skew_symmetric_backward_kernel_working[grid](
            grad_output,
            grad_vec,
            batch_size,
            N,
            stride_mat_batch=grad_output.stride(0),
            stride_mat_row=grad_output.stride(1),
            stride_mat_col=grad_output.stride(2),
            stride_vec_batch=grad_vec.stride(0),
            stride_vec_element=grad_vec.stride(1),
            # BLOCK_SIZE=BLOCK_SIZE,
        )
        return grad_vec, None
    

def _cayley_batch(Q: torch.Tensor, block_size: int, num_neumann_terms: int = 5) -> torch.Tensor:
    # ... (rest of _cayley_batch remains the same)
    b, n_elements = Q.shape
    expected_elements = block_size * (block_size - 1) // 2
    if n_elements != expected_elements:
        raise ValueError(f"Input Q has {n_elements} elements, but expected {expected_elements} for block_size={block_size}")

    Q_skew = SkewSymmetric.apply(Q, block_size) # (b, block_size, block_size)

    R = torch.eye(block_size, device=Q.device, dtype=Q.dtype).repeat(b, 1, 1)
    I_batch = R.clone()

    if num_neumann_terms > 1:
        R.add_(Q_skew, alpha=2.0)
        if num_neumann_terms > 2:
            Q_squared = torch.bmm(Q_skew, Q_skew)
            R.add_(Q_squared, alpha=2.0)

            Q_power = Q_squared
            for i in range(3, num_neumann_terms):
                Q_power = torch.bmm(Q_power, Q_skew)
                R.add_(Q_power, alpha=2.0)

    return R


def _block_diagonal(blocks: torch.Tensor, rank: int) -> torch.Tensor:
    # ... (rest of _block_diagonal remains the same)
    if blocks.shape[0] != rank:
         raise ValueError(f"Number of blocks {blocks.shape[0]} does not match rank {rank}")
    A = torch.block_diag(*[blocks[i, ...] for i in range(rank)])
    return A


# --- Test Function ---
def test_oft_forward_equivalence(B, N, r, b, dtype, device='cuda'):
    print(f"\n--- Testing Forward Equivalence (Self-Contained, Original Dtype) (B={B}, N={N}, r={r}, b={b}, dtype={dtype}, device={device}) ---")
    in_features = r * b
    out_features = max(1, in_features // 2) # Ensure out_features >= 1

    torch.manual_seed(42)
    # Initialize base layer in the target dtype
    base_layer = nn.Linear(in_features, out_features, bias=True).to(device).to(dtype)

    # Generate oft_r and input in the target dtype
    oft_r = torch.randn(r, b * (b - 1) // 2, device=device, dtype=dtype) * 0.01 # Small values often used
    x = torch.randn(B, N, in_features, device=device, dtype=dtype, requires_grad=False)

    # Option 1: (W @ R^T) @ x (calculation in original dtype)
    print("Calculating Option 1: (W @ R^T) @ x")
    with torch.no_grad():
        orth_rotate_batch = _cayley_batch(oft_r, b) # R blocks
        R_full = _block_diagonal(orth_rotate_batch, r) # R

        W1 = base_layer.weight.clone()
        bias1 = base_layer.bias.clone() if base_layer.bias is not None else None
        x1 = x.clone()

        # Compute W @ R^T
        W_transposed = W1.transpose(0, 1) # W^T
        rotated_weight_intermediate = torch.matmul(R_full, W_transposed) # R @ W^T
        rotated_weight = rotated_weight_intermediate.transpose(0, 1) # (R @ W^T)^T = W @ R^T

        # Compute y1 = x @ (W @ R^T)^T + bias = x @ R @ W^T + bias
        y1_flat = F.linear(x1, rotated_weight, bias1) # bias1 automatically handled if None
        y1 = y1_flat.view(B, N, out_features)

    print("Option 1 Calculation Done.")

    # Option 2: (x @ R) @ W^T + bias (calculation in original dtype)
    print("Calculating Option 2: (x @ R) @ W^T")
    with torch.no_grad():
        W2 = base_layer.weight.clone()
        bias2 = base_layer.bias.clone() if base_layer.bias is not None else None
        x2 = x.clone()

        # Recompute R for Option 2 path
        orth_rotate = _cayley_batch(oft_r, b)
        '''
        R_full2 = _block_diagonal(orth_rotate, r) # R

        # Ensure intermediate R matrices are the same (debug check)
        assert torch.allclose(R_full, R_full2, atol=1e-6), "R_full matrices differ between Option 1 and 2"

        # Compute x_rotated = x @ R
        # Use torch.matmul for: (B, N, in) @ (in, in) -> (B, N, in)
        x_rotated = torch.matmul(x2, R_full2)

        # Compute y2 = (x @ R) @ W^T + bias
        y2 = F.linear(x_rotated, W2, bias2) # bias2 automatically handled if None
        '''

        batch_dims = x2.shape[:-1]
        x_reshaped = x2.view(*batch_dims, r, -1)
        x_rotated_reshaped = torch.einsum('...rk,rkc->...rc', x_reshaped, orth_rotate)
        x_rotated = x_rotated_reshaped.reshape(*batch_dims, in_features)
        y2 = F.linear(x_rotated, W2, bias2)


    # Option 2: (x @ R) @ W^T + bias (calculation in original dtype)
    print("Option 2 Calculation Done.")

    # Compare Results directly in the original dtype
    print("Comparing results (original dtype)...")
    y1_comp = y1 # No conversion
    y2_comp = y2 # No conversion

    # Use tolerances appropriate for the original dtype
    if dtype == torch.float32:
        rtol = 1e-5
        atol = 1e-6
    elif dtype == torch.bfloat16:
        # These might need adjustment depending on the scale of values and operations
        rtol = 1e-2
        atol = 1e-2
    else: # float16
        rtol = 1e-3
        atol = 1e-3

    are_close = torch.allclose(y1_comp, y2_comp, rtol=rtol, atol=atol)

    if are_close:
        print(f"✅ Test Passed: Outputs are close within tolerance (rtol={rtol}, atol={atol}) using dtype {dtype}.")
    else:
        print(f"❌ Test Failed: Outputs differ significantly using dtype {dtype}.")
        # Compute differences in float32 for more informative printout
        diff = torch.abs(y1.to(torch.float32) - y2.to(torch.float32))
        max_abs_diff = torch.max(diff).item()
        # Use y2 for relative diff calculation stability
        max_rel_diff = torch.max(diff / (torch.abs(y2.to(torch.float32)) + atol)).item()
        print(f"   Max absolute difference (f32): {max_abs_diff:.6e}")
        print(f"   Max relative difference (f32): {max_rel_diff:.6e}")
        print(f"   Note: Failure might be expected due to precision limits of {dtype} and different operation orders.")

    print("-" * 40)
    return are_close

# --- Main Execution Block ---
# (Keep the main block as it was)
if __name__ == '__main__':
    # --- Configuration ---
    B = 4
    N = 16
    r = 8
    b = 32 # Block size must be >= 2 for skew-symmetric matrices
    dtype = torch.bfloat16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ---------------------

    print(f"Using device: {device}")
    in_features = r * b
    if b < 2:
        print(f"Warning: Block size b={b} is less than 2. SkewSymmetric requires b>=2. Adjusting b=2.")
        b = 2
        in_features = r * b
    if in_features == 0:
         raise ValueError("Calculated in_features is 0. Check r and b.")

    print(f"Test Params: B={B}, N={N}, r={r}, b={b} (in_features={in_features}), dtype={dtype}")
    print("="*40)

    test_oft_forward_equivalence(B, N, r, b, dtype=dtype, device=device)
    print("="*40)

    print("Script finished.")