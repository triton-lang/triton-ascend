"""FA accuracy test: CVSplit/DCVP/plain baseline vs PyTorch reference.
Config: B=1, H=1, N=8192, D=64, BLOCK_M=32, BLOCK_N=32 (large-context inner-loop scale)
  Inner loop: N_CTX/BLOCK_N = 8192/32 = 256 iterations (64 after unroll by 4)
  ISOLATED: only ACTIVE_BLOCKS=1 query tile launched -> 1 AI core (its 2 veccores),
  so sim cost stays ~N=512 level while the inner loop runs the full 64 post-unroll
  iters over the 8192-token context. Accuracy is checked on the computed tile only.
"""
import os

if __name__ == "__main__":
    # Configure the standalone execution environment before importing the NPU
    # runtime, without changing the caller's environment when this module is
    # imported for discovery or reuse.
    os.environ["TRITON_ASCEND_SOC_VERSION"] = "Ascend950PR_9589"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
import torch_npu
import triton
import triton.language as tl
import math

# Config matched to test_fa_reference_scoped.py (the manually-written target kernel):
#   Z=1, H=1, N_CTX=256, HEAD_DIM=64, BM=BN=32, non-causal, std=0.5 init, seed 42
B, H, N, D = 1, 1, 8192, 64
# Block sizes are env-overridable so we can test our pass at the target's native
# geometry (BLOCK_M=BLOCK_N=128 -> ROW_SPLIT 64 rows/veccore, [1,64] row slices),
# where the per-row SIMD vector path is register-aligned. Default stays 32x32.
BLOCK_M = int(os.environ.get("FA_BLOCK_M", "32"))
BLOCK_N = int(os.environ.get("FA_BLOCK_N", "32"))
CORE_NUM = 32
# Isolate the inner loop: launch only this many query tiles (1 -> 1 AI core / 2
# veccores). Keeps sim cost ~constant while the inner loop scales with N_CTX.
ACTIVE_BLOCKS = 1
sm_scale = 1.0 / math.sqrt(D)


def reference_attention(q, k, v, sm_scale):
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    attn = torch.matmul(q_f, k_f.transpose(-2, -1)) * sm_scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v_f).half()


# Keep the canonical Flash-Attention helper signature, including the currently
# unused start_m/STAGE/offset parameters. This regression intentionally matches
# the production frontend shape used to generate and compare the manual and
# CV-split compiler schedules; removing them would change that test contract.
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale: tl.constexpr, BLOCK_M: tl.constexpr,
                    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr, offs_m: tl.constexpr,
                    offs_n: tl.constexpr, N_CTX: tl.constexpr):
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        qk = qk * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        qk = qk - m_ij[:, None]
        p = tl.math.exp(qk)
        p_cast = p.to(q.type)
        v = tl.load(V_block_ptr)
        pv = tl.dot(p_cast, v)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + pv
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, sm_scale: tl.constexpr, M, Out, stride_qz: tl.constexpr, stride_qh: tl.constexpr,
              stride_qm: tl.constexpr, stride_qk: tl.constexpr, stride_kz: tl.constexpr, stride_kh: tl.constexpr,
              stride_kn: tl.constexpr, stride_kk: tl.constexpr, stride_vz: tl.constexpr, stride_vh: tl.constexpr,
              stride_vn: tl.constexpr, stride_vk: tl.constexpr, stride_oz: tl.constexpr, stride_oh: tl.constexpr,
              stride_om: tl.constexpr, stride_on: tl.constexpr, Z: tl.constexpr, H: tl.constexpr, N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr,
              NUM_BLOCKS_M: tl.constexpr, NUM_BLOCKS: tl.constexpr, AICORE_NUM: tl.constexpr):
    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, AICORE_NUM):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H

        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh

        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + q_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        q = tl.load(Q_block_ptr)

        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, task_m_idx, sm_scale, BLOCK_M,
                                        HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX)

        acc = acc / l_i[:, None]
        m_i += tl.math.log(l_i)
        m_ptrs = M + task_hz_idx * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def get_compile_options(variant, unroll_factor=4):
    """Return an explicit compiler policy for a fair three-way comparison.

    A5 enables DynamicCVPipeline by default, so an empty kwargs dictionary is
    not a plain baseline.  Keep both feature switches explicit here to make it
    impossible for a default change to silently alter the comparison.
    """
    if variant == "default":
        return {}
    if variant == "baseline":
        return {
            "enable_dynamic_cv_pipeline": False,
            "enable_cv_split_scheduling": False,
        }
    if variant == "dcvp":
        return {
            "enable_dynamic_cv_pipeline": True,
            "enable_cv_split_scheduling": False,
        }
    if variant == "cvsplit":
        return {
            "enable_dynamic_cv_pipeline": False,
            "enable_cv_split_scheduling": True,
            "cv_split_unroll_factor": unroll_factor,
        }
    if variant == "auto":
        return {
            "enable_dynamic_cv_pipeline": True,
            "enable_cv_split_scheduling": True,
            "cv_split_unroll_factor": unroll_factor,
        }
    raise ValueError(
        f"variant must be default|baseline|dcvp|cvsplit|auto, got {variant!r}")


def run_attention(q, k, v, sm_scale, use_cvsplit=False, variant=None):
    o = torch.empty_like(q)
    NUM_BLOCKS_M = triton.cdiv(q.shape[2], BLOCK_M)
    # Cap the number of query tiles actually launched (isolate the inner loop).
    NUM_BLOCKS = min(NUM_BLOCKS_M * q.shape[0] * q.shape[1], ACTIVE_BLOCKS)
    grid = (CORE_NUM, )
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    if variant is None:
        variant = "cvsplit" if use_cvsplit else "baseline"
    kwargs = get_compile_options(
        variant, int(os.environ.get("CV_SPLIT_UNROLL", "4")))

    _attn_fwd[grid](q, k, v, sm_scale, M, o, q.stride(0),
                    q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3), o.stride(0), o.stride(1), o.stride(2),
                    o.stride(3), q.shape[0], q.shape[1], N_CTX=q.shape[2], HEAD_DIM=q.shape[-1], BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N, STAGE=0, NUM_BLOCKS_M=NUM_BLOCKS_M, NUM_BLOCKS=NUM_BLOCKS, AICORE_NUM=CORE_NUM,
                    **kwargs)
    return o


def check_accuracy(name, out, ref, atol=0.05, rtol=0.05):
    out_cpu = out.cpu().float()
    ref_f = ref.float()
    max_diff = (out_cpu - ref_f).abs().max().item()
    mean_diff = (out_cpu - ref_f).abs().mean().item()
    out_flat = out_cpu.flatten()
    ref_flat = ref_f.flatten()
    if out_flat.norm() < 1e-8 or ref_flat.norm() < 1e-8:
        cos_sim = 0.0
        print(f"  WARNING: output or ref is near-zero (norms: out={out_flat.norm():.6f}, ref={ref_flat.norm():.6f})")
    else:
        cos_sim = torch.nn.functional.cosine_similarity(out_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item()
    close = torch.allclose(out_cpu, ref_f, atol=atol, rtol=rtol)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Output range:  [{out_cpu.min():.4f}, {out_cpu.max():.4f}]")
    print(f"  Ref range:     [{ref_f.min():.4f}, {ref_f.max():.4f}]")
    print(f"  Max abs diff:  {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")
    print(f"  Cosine sim:    {cos_sim:.8f}")
    print(f"  allclose(atol={atol}, rtol={rtol}): {close}")
    print(f"  PASS" if close else f"  FAIL (max_diff={max_diff:.6f})")
    return close


def main():
    variant = os.environ.get("FA_VARIANT", "cvsplit").lower()
    if variant not in ("default", "cvsplit", "dcvp", "baseline", "auto"):
        raise ValueError(
            f"FA_VARIANT must be default|cvsplit|dcvp|baseline|auto, got {variant!r}")

    torch.manual_seed(42)
    q_cpu = torch.empty(B, H, N, D, dtype=torch.float16).normal_(mean=0.0, std=0.5)
    k_cpu = torch.empty(B, H, N, D, dtype=torch.float16).normal_(mean=0.0, std=0.5)
    v_cpu = torch.empty(B, H, N, D, dtype=torch.float16).normal_(mean=0.0, std=0.5)

    # Only the first ACTIVE_BLOCKS*BLOCK_M query rows are computed by the
    # kernel, so compute the reference for only those rows.
    ref_rows = ACTIVE_BLOCKS * BLOCK_M
    ref_out = reference_attention(q_cpu[:, :, :ref_rows, :], k_cpu, v_cpu, sm_scale)
    print(f"Reference output: shape={ref_out.shape}, mean={ref_out.float().mean():.6f}, "
          f"std={ref_out.float().std():.6f}")
    print(f"Reference range: [{ref_out.float().min():.4f}, {ref_out.float().max():.4f}]")

    q_npu = q_cpu.to("npu")
    k_npu = k_cpu.to("npu")
    v_npu = v_cpu.to("npu")

    num_blocks_m = triton.cdiv(N, BLOCK_M)
    num_blocks = min(num_blocks_m * B * H, ACTIVE_BLOCKS)
    print(f"\nConfig: B={B}, H={H}, N={N}, D={D}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print(f"  Inner loop iters (pre-unroll): {N // BLOCK_N}")
    unroll_factor = int(os.environ.get("CV_SPLIT_UNROLL", "4"))
    print(f"  Inner loop iters (post-unroll by {unroll_factor}): {N // BLOCK_N // unroll_factor}")
    print(f"  ACTIVE query tiles: {num_blocks} (-> {num_blocks} AI core(s), each 2 veccores)")
    print(f"  NUM_BLOCKS={num_blocks}, CORE_NUM={CORE_NUM}")

    # Run baseline and CV-split in separate invocations so each has isolated
    # profiling and accuracy state.
    use_cvsplit = variant == "cvsplit"
    names = {
        "default": "COMPILER DEFAULT POLICY",
        "cvsplit": "CVSPLIT (our pass)",
        "auto": "CVSPLIT TRY WITH DYNAMIC CV FALLBACK",
        "dcvp": "DYNAMIC CV PIPELINE",
        "baseline": "PLAIN BASELINE (both CV pipelines disabled)",
    }
    name = names[variant]

    print("\n" + "=" * 60)
    print(f" Running {name}...  [FA_VARIANT={variant}]")
    print("=" * 60)
    try:
        out = run_attention(
            q_npu, k_npu, v_npu, sm_scale,
            use_cvsplit=use_cvsplit, variant=variant)
        ok = check_accuracy(variant.upper(), out[:, :, :ref_rows, :], ref_out)
    except Exception as exc:
        print(f"  {variant.upper()} FAILED: {exc}")
        ok = False

    print("\n" + "=" * 60)
    print(f" SUMMARY: {variant}={'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    print("\nACCURACY TEST DONE")


if __name__ == "__main__":
    main()
