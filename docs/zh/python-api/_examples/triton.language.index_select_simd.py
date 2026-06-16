import triton
import triton.language as tl
import triton.language.extra.ascend.libdevice as libdevice


@triton.jit
def embedding_kernel(
    embed_ptr,  # [vocab_size, embed_dim]
    indices_ptr,  # [batch_size]
    output_ptr,  # [batch_size, embed_dim]
    vocab_size: tl.constexpr,
    embed_dim: tl.constexpr,
):
    pid = tl.program_id(0)

    # 加载索引
    indices = tl.load(indices_ptr + pid * 16 + tl.arange(0, 16))

    # 使用 index_select 批量读取嵌入向量
    embeddings = libdevice.index_select_simd(src=embed_ptr, dim=0, index=indices, src_shape=(vocab_size, embed_dim),
                                             src_offset=(-1, 0), read_shape=(-1, embed_dim))

    # 存储结果
    offsets = tl.arange(0, 16)[:, None] * embed_dim + tl.arange(0, embed_dim)[None, :]
    tl.store(output_ptr + pid * 16 * embed_dim + offsets, embeddings)
