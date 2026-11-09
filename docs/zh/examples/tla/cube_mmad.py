import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def cube_mmad_example(A_ptr, B_ptr, O_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    a = tla.view(A_ptr, (M, K), strides=(K, 1))
    b = tla.view(B_ptr, (K, N), strides=(N, 1))
    o = tla.view(O_ptr, (M, N), strides=(N, 1))

    # A/B 放在 L1，累加器放在 L0C。
    a_l1 = tla.alloc_buffer((M, K), tl.float16, space="l1")
    b_l1 = tla.alloc_buffer((K, N), tl.float16, space="l1")
    acc = tla.alloc_buffer((M, N), tl.float32, space="l0c")

    # GM -> L1（ND -> NZ 布局转换）。
    tla.load_gm_to_l1(a, a_l1)
    tla.load_gm_to_l1(b, b_l1)

    # C = A x B + C。
    tla.cube.mmad(a_l1, b_l1, acc, transpose_b=False, clear_accum=False)

    # L0C -> GM（Fixpipe）。
    tla.store_l0c_to_gm(acc, o, tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND))
