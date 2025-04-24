import pytest
import torch

import triton
import triton.ops
from matmul_qoft import matmul


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def sparsify_tensor(x, mask, block):
    ret = torch.empty((mask.sum(), block, block), dtype=x.dtype, device=x.device)
    for idx, (h, i, j) in enumerate(zip(*mask.nonzero(as_tuple=True))):
        ret[idx, :, :] = x[h, i * block:(i + 1) * block, j * block:(j + 1) * block]
    return ret


def make_pair(shape, device="cuda", alpha=1e-2, beta=0., trans=False, data=None, dtype=torch.float32):
    if data is None:
        data = torch.randn(shape, dtype=torch.float32, requires_grad=True, device=device)
    ref_ret = data
    ref_ret = ref_ret * alpha + beta
    ref_ret = ref_ret.half().to(dtype)
    if trans:
        ref_ret = ref_ret.t().requires_grad_()
    ref_ret = ref_ret.detach().requires_grad_()
    tri_ret = ref_ret.clone().detach().requires_grad_()
    return ref_ret, tri_ret


def mask_tensor(x, mask, block, value=0):
    ret = x.clone()
    for h, i, j in zip(*(mask == 0).nonzero(as_tuple=True)):
        ret[h, i * block:(i + 1) * block, j * block:(j + 1) * block] = value
    return ret


def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, device, M=512, N=384, K=512):
    # A: M x K
    # B: K x N
    # C: M x N
    seed = 0
    torch.manual_seed(seed)
    is_qoft_sdd = MODE == "qoft_sdd"
    is_qoft_dsd = MODE == "qoft_dsd"
    is_qoft_dds = MODE == "qoft_dds"
    do_sparsify = lambda x: sparsify_tensor(x, layout, BLOCK)
    do_mask = lambda x: mask_tensor(x, layout, BLOCK)
    # create inputs
    # create op
    if is_qoft_dsd: # a_shape[2] == a_shape[3]
        # a_shape = (H, K, K) if TRANS_A else (H, K, K)
        H = K // BLOCK
        a_shape = (H, BLOCK, BLOCK)
        b_shape = (1, N, K) if TRANS_B else (1, K, N)
        c_shape = (1, K, N)

    elif is_qoft_dds: # b_shape[2] == b_shape[3]
        H = K // BLOCK
        a_shape = (1, K, M) if TRANS_A else (1, M, K)
        # b_shape = (H, K, K) if TRANS_B else (H, K, K)
        b_shape = (H, BLOCK, BLOCK)
        c_shape = (1, M, K)

    else:
        a_shape = (1, K, M) if TRANS_A else (1, M, K)
        b_shape = (1, N, K) if TRANS_B else (1, K, N)
        c_shape = (1, M, N)

    shape = {
        "qoft_sdd": (M, N),
        "qoft_dsd": (a_shape[-2], a_shape[-1]),
        "qoft_dds": (b_shape[-2], b_shape[-1]),
    }[MODE]
    # layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
    layout = torch.eye(H).unsqueeze(0).to(torch.int64)
    # create data
    a_ref, a_tri = make_pair(a_shape, alpha=.1, dtype=DTYPE)
    b_ref, b_tri = make_pair(b_shape, alpha=.1, dtype=DTYPE)
    dc_ref, dc_tri = make_pair(c_shape, dtype=DTYPE)
    # compute [torch]
    dc_ref = do_mask(dc_ref) if is_qoft_sdd else dc_ref
    a_ref = torch.block_diag(*do_mask(a_ref)).view(1, a_shape[0] * a_shape[1], a_shape[0] * a_shape[1]) if is_qoft_dsd else a_ref
    b_ref = torch.block_diag(*do_mask(b_ref)).view(1, b_shape[0] * b_shape[1], b_shape[0] * b_shape[1]) if is_qoft_dds else b_ref
    a_ref.retain_grad()
    b_ref.retain_grad()
    c_ref = torch.matmul(a_ref.transpose(-2, -1) if TRANS_A else a_ref, b_ref.transpose(-2, -1) if TRANS_B else b_ref)
    c_ref.backward(dc_ref)
    c_ref = do_sparsify(c_ref) if is_qoft_sdd else c_ref
    da_ref = do_sparsify(a_ref.grad) if is_qoft_dsd else a_ref.grad
    db_ref = do_sparsify(b_ref.grad) if is_qoft_dds else b_ref.grad
    # triton result
    dc_tri = do_sparsify(dc_tri) if is_qoft_sdd else dc_tri
    # a_tri = do_sparsify(a_tri) if is_qoft_dsd else a_tri
    # b_tri = do_sparsify(b_tri) if is_qoft_dds else b_tri
    a_tri.retain_grad()
    b_tri.retain_grad()
    breakpoint()
    op = matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device=device)
    c_tri = op(a_tri, b_tri)
    c_tri.backward(dc_tri)
    da_tri = a_tri.grad
    db_tri = b_tri.grad

    # Bigger tolerance for AMD MI200 devices.
    # MI200 devices use reduced precision fp16 and bf16 and flush input and
    # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    tol = {'atol': 1e-3, 'rtol': 0} if is_hip_mi200() else {}

    # compare
    torch.testing.assert_close(c_ref, c_tri, **tol)
    torch.testing.assert_close(da_ref, da_tri, **tol)
    torch.testing.assert_close(db_ref, db_tri, **tol)

    gpu_memory_usage_a_tri = a_tri.element_size() * a_tri.nelement()
    gpu_memory_usage_a_ref = a_ref.element_size() * a_ref.nelement()
    print(f"GPU memory usage for a_tri: {gpu_memory_usage_a_tri} bytes")
    print(f"GPU memory usage for a_ref: {gpu_memory_usage_a_ref} bytes")

    gpu_memory_usage_da_tri = a_tri.grad.element_size() * a_tri.grad.nelement()
    gpu_memory_usage_da_ref = a_ref.grad.element_size() * a_ref.grad.nelement()
    print(f"GPU memory usage for da_tri: {gpu_memory_usage_da_tri} bytes")
    print(f"GPU memory usage for da_ref: {gpu_memory_usage_da_ref} bytes")

    breakpoint()


    # Timing the operations
    print("\n--- Timing ---")
    ms_tri = triton.testing.do_bench(lambda: op(a_tri, b_tri), rep=50)
    print(f"Triton op time: {ms_tri:.4f} ms")

    ms_ref = triton.testing.do_bench(lambda: torch.matmul(a_ref.transpose(-2, -1) if TRANS_A else a_ref, b_ref.transpose(-2, -1) if TRANS_B else b_ref), rep=50)
    print(f"Torch matmul time: {ms_ref:.4f} ms")
    print("--------------\n")

    exit()

    # compare
    print("--------------------------------------------------------------")
    perf = lambda ms: 2 * M * N * K * H * 1e9 / ( ms * 1e-3)
    total_op = 2 * M * N * K * H

    print('''MODE={}, H={}, M={}, N={}, K={}, total_op={}. '''
            .format(MODE, H, M, N, K, total_op))

    diff = torch.sum(torch.abs(c_ref - c_tri))
    print('total diff = {:.10f}'.format(diff))

    if(torch.allclose(c_ref, c_tri, atol = 1e-5, rtol = 0)):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

    print(a_tri.shape, b_tri.shape)
    print(a_ref.shape, b_ref.shape)

    # Triton performance and memory profiling
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    ms = triton.testing.do_bench(lambda: op(a_tri, b_tri), rep=20)
    triton_memory_alloc = torch.cuda.memory_allocated(device)
    triton_peak_memory = torch.cuda.max_memory_allocated(device)

    print('''Triton: GFLOPS: {:.3f}, time: {:.6f}ms, memory: {:.2f}MB, peak memory: {:.2f}MB.'''.format(
        perf(ms), ms, triton_memory_alloc / 1e6, triton_peak_memory / 1e6))

    # PyTorch performance and memory profiling
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    ms_torch = triton.testing.do_bench(
        lambda: torch.matmul(a_ref.transpose(2, 3) if TRANS_A else a_ref,
                             b_ref.transpose(2, 3) if TRANS_B else b_ref),
        rep=20
    )
    torch_memory_alloc = torch.cuda.memory_allocated(device)
    torch_peak_memory = torch.cuda.max_memory_allocated(device)

    print('''Torch: GFLOPS: {:.3f}, time: {:.6f}ms, memory: {:.2f}MB, peak memory: {:.2f}MB.'''.format(
        perf(ms_torch), ms_torch, torch_memory_alloc / 1e6, torch_peak_memory / 1e6))


if __name__ == "__main__":
    test_matmul("qoft_dsd", False, False, 32, torch.float32, "cuda", M=2048, N=2048, K=2048) # M=512, N=384, K=4096
    # test_matmul("qoft_dsd", False, False, 256, torch.bfloat16, "cuda", M=4096, N=4096, K=4096) # M=512, N=384, K=4096