import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def sync_example(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
    a = tla.view(A_ptr, (N, ), strides=(1, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ))
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    tla.copy(a, a_ub)
    tla.copy(b, b_ub)

    # 生产者 MTE2 -> 消费者 V，事件 ID 范围为 [0, 7]。
    tla.sync.set_flag("MTE2", "V", 0)
    tla.sync.wait_flag("MTE2", "V", 0)

    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0])
        bv = tla.simd.vlds(b_ub[0])
        tla.simd.vsts(o_ub[0], tla.simd.vadd(av, bv), dist="NORM_B32")

    # 等待指定流水上的前序指令完成。
    tla.sync.barrier("ALL")
    tla.sync.set_flag("V", "MTE3", 1)
    tla.sync.wait_flag("V", "MTE3", 1)
    tla.copy(o_ub, o)

    # 核间 flag：flag 范围 [0, 15]。
    tla.sync.block_set_flag(0, "V", 3)
    tla.sync.block_wait_flag("V", 3)

    # False 只同步 AIV，True 同步 AIC 和 AIV。
    tla.sync.sync_all(False)
