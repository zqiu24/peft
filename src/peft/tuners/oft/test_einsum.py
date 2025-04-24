import torch
import time
import math

# --- Implementation 1: Einsum ---
def benchmark_input_rotation_einsum(B, N, r, b, dtype=torch.float32, num_runs=100, warmup_runs=10):
    """Benchmarks input rotation using torch.einsum on CUDA."""
    device = 'cuda'
    if not torch.cuda.is_available():
         raise RuntimeError("This benchmark requires CUDA to be available.")

    in_features = r * b
    x_shape = (B, N, in_features) if N > 1 else (B, in_features)
    x = torch.randn(*x_shape, dtype=dtype, device=device)
    oft_rotation = torch.randn(r, b, b, dtype=dtype, device=device)

    def rotate_input_einsum(inp, rotation_matrix, rank, block_size):
        rot_dtype = rotation_matrix.dtype
        inp = inp.to(rot_dtype)
        batch_dims = inp.shape[:-1]
        in_feats = rank * block_size
        x_reshaped = inp.view(*batch_dims, rank, block_size)
        # Einsum: 'rck,...rk->...rc' (matrix @ vector for each block)
        x_rotated_reshaped = torch.einsum('rck,...rk->...rc', rotation_matrix, x_reshaped)
        x_rotated = x_rotated_reshaped.reshape(*batch_dims, in_feats)
        return x_rotated

    # --- Timing Logic (Identical for all benchmarks) ---
    print(f"Method: Einsum - Warming up for {warmup_runs} runs...")
    for _ in range(warmup_runs):
        _ = rotate_input_einsum(x, oft_rotation, r, b)
    torch.cuda.synchronize()

    print(f"Method: Einsum - Starting timing for {num_runs} runs...")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        start_events[i].record()
        _ = rotate_input_einsum(x, oft_rotation, r, b)
        end_events[i].record()
    torch.cuda.synchronize()

    total_gpu_time_ms = sum(start_events[i].elapsed_time(end_events[i]) for i in range(num_runs))
    avg_time_ms = total_gpu_time_ms / num_runs

    print(f"\n--- [Einsum] CUDA Benchmark Results ---")
    print(f"Input shape (x): {tuple(x_shape)}")
    print(f"Rotation shape (R): {tuple(oft_rotation.shape)}")
    print(f"Dtype: {dtype}, Num runs: {num_runs}")
    print(f"Average time per run: {avg_time_ms:.6f} ms")
    print(f"--------------------------------------")
    return avg_time_ms

# --- Implementation 2: BMM (Placeholder) ---
def benchmark_input_rotation_einsum_flattened(B, N, r, b, dtype=torch.float32, num_runs=100, warmup_runs=10):
    """Benchmarks input rotation using torch.bmm (Placeholder) on CUDA."""
    device = 'cuda'
    if not torch.cuda.is_available():
         raise RuntimeError("This benchmark requires CUDA to be available.")

    in_features = r * b
    x_shape = (B, N, in_features) if N > 1 else (B, in_features)
    x = torch.randn(*x_shape, dtype=dtype, device=device)
    oft_rotation = torch.randn(r, b, b, dtype=dtype, device=device) # Shape (r, b, b)

    # --- Placeholder BMM Core Logic ---
    def rotate_input_bmm(inp, rotation_matrix, rank, block_size):
        B, N, in_features = inp.shape
        b, r, _ = rotation_matrix.shape
        assert in_features == b * r, "in_features must be equal to b * r"
        
        # Reshape x to separate the blocks: (B*N, b, r)
        x_reshaped = x.view(B * N, b, r)
        
        # Apply block-wise rotation using einsum: (B*N, b, r) x (b, r, r) -> (B*N, b, r)
        x_rotated = torch.einsum('nbi,bio->nbo', x_reshaped, rotation_matrix)
        
        # Reshape back to original dimensions
        x_rotated = x_rotated.reshape(B, N, -1)  # -1 infers b*r = in_features
        
        return x_rotated
    # --- End Placeholder BMM Core Logic ---

    # --- Timing Logic ---
    print(f"Method: BMM - Warming up for {warmup_runs} runs...")
    for _ in range(warmup_runs):
        _ = rotate_input_bmm(x, oft_rotation, r, b)
    torch.cuda.synchronize()

    print(f"Method: BMM - Starting timing for {num_runs} runs...")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        start_events[i].record()
        _ = rotate_input_bmm(x, oft_rotation, r, b)
        end_events[i].record()
    torch.cuda.synchronize()

    total_gpu_time_ms = sum(start_events[i].elapsed_time(end_events[i]) for i in range(num_runs))
    avg_time_ms = total_gpu_time_ms / num_runs

    print(f"\n--- [EINSUM-FLATTENED] CUDA Benchmark Results ---")
    print(f"Input shape (x): {tuple(x_shape)}")
    print(f"Rotation shape (R): {tuple(oft_rotation.shape)}")
    print(f"Dtype: {dtype}, Num runs: {num_runs}")
    print(f"Average time per run: {avg_time_ms:.6f} ms")
    print(f"-----------------------------------")
    return avg_time_ms


# --- Implementation 3: Manual/Other (Placeholder) ---
def benchmark_input_rotation_bmm(B, N, r, b, dtype=torch.float32, num_runs=100, warmup_runs=10):
    """Benchmarks input rotation using a manual method (Placeholder) on CUDA."""
    device = 'cuda'
    if not torch.cuda.is_available():
         raise RuntimeError("This benchmark requires CUDA to be available.")

    in_features = r * b
    x_shape = (B, N, in_features) if N > 1 else (B, in_features)
    x = torch.randn(*x_shape, dtype=dtype, device=device)
    oft_rotation = torch.randn(r, b, b, dtype=dtype, device=device)

    # --- Placeholder Manual Core Logic ---
    def rotate_input_manual(inp, rotation_matrix, rank, block_size):
        rot_dtype = rotation_matrix.dtype
        inp = inp.to(rot_dtype)
        batch_dims = inp.shape[:-1]
        in_feats = rank * block_size
        x_reshaped = inp.view(*batch_dims, rank, block_size)
        # Replace this einsum with your logic
        x_rotated_reshaped = torch.einsum('rck,...rk->...rc', rotation_matrix, x_reshaped)
        x_rotated = x_rotated_reshaped.reshape(*batch_dims, in_feats)
        return x_rotated
    # --- End Placeholder Manual Core Logic ---

    # --- Timing Logic ---
    print(f"Method: Manual - Warming up for {warmup_runs} runs...")
    for _ in range(warmup_runs):
        _ = rotate_input_manual(x, oft_rotation, r, b)
    torch.cuda.synchronize()

    print(f"Method: Manual - Starting timing for {num_runs} runs...")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        start_events[i].record()
        _ = rotate_input_manual(x, oft_rotation, r, b)
        end_events[i].record()
    torch.cuda.synchronize()

    total_gpu_time_ms = sum(start_events[i].elapsed_time(end_events[i]) for i in range(num_runs))
    avg_time_ms = total_gpu_time_ms / num_runs

    print(f"\n--- [BMM] CUDA Benchmark Results ---")
    print(f"Input shape (x): {tuple(x_shape)}")
    print(f"Rotation shape (R): {tuple(oft_rotation.shape)}")
    print(f"Dtype: {dtype}, Num runs: {num_runs}")
    print(f"Average time per run: {avg_time_ms:.6f} ms")
    print(f"--------------------------------------")
    return avg_time_ms


if __name__ == '__main__':
    # --- Configuration ---
    B = 32         # Batch size
    N = 512        # Sequence length (e.g., tokens) set N=1 for simple (B, in_features)
    r = 128        # Rank (number of blocks)
    b = 32         # Block size
    dtype = torch.bfloat16 # Use float32 or bfloat16/float16
    num_runs_main = 200 # Number of timed runs
    warmup_runs_main = 20 # Number of warmup runs
    # ---------------------

    # Derived parameter
    in_features = r * b
    print(f"Running CUDA benchmarks with B={B}, N={N}, r={r}, b={b} (in_features={in_features}), dtype={dtype}")
    print("="*40)

    # Call the benchmark functions
    benchmark_input_rotation_einsum(B, N, r, b, dtype=dtype, num_runs=num_runs_main, warmup_runs=warmup_runs_main)
    print("="*40)
    benchmark_input_rotation_einsum_flattened(B, N, r, b, dtype=dtype, num_runs=num_runs_main, warmup_runs=warmup_runs_main)
    print("="*40)
    benchmark_input_rotation_bmm(B, N, r, b, dtype=dtype, num_runs=num_runs_main, warmup_runs=warmup_runs_main)
    print("="*40)
