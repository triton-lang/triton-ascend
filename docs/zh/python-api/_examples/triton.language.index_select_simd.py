import torch
import triton
import triton.language as tl
from triton.language.extra.cann.extension import index_select_simd


@triton.jit
def embedding_kernel(
    embed_ptr,  # [vocab_size, embed_dim]
    indices_ptr,  # [batch_size]
    output_ptr,  # [batch_size, embed_dim]
    vocab_size: tl.constexpr,
    embed_dim: tl.constexpr,
    batch_size: tl.constexpr,
):
    indices = tl.load(indices_ptr + tl.arange(0, batch_size))
    embeddings = index_select_simd(src=embed_ptr, dim=0, index=indices, src_shape=(vocab_size, embed_dim),
                                   src_offset=(-1, 0), read_shape=(-1, embed_dim))
    offsets = tl.arange(0, batch_size)[:, None] * embed_dim + tl.arange(0, embed_dim)[None, :]
    tl.store(output_ptr + offsets, embeddings)


def test_index_select():
    vocab_size, embed_dim = 10, 8
    batch_size = 5
    torch.manual_seed(0)
    embed = torch.randn(vocab_size, embed_dim, dtype=torch.float32).npu()
    indices = torch.randint(0, vocab_size, (batch_size, ), dtype=torch.int64).npu()
    output = torch.empty((batch_size, embed_dim), dtype=torch.float32, device="npu")

    embedding_kernel[(1, )](embed, indices, output, vocab_size, embed_dim, batch_size)

    out_cpu = output.cpu()
    ref = embed.cpu()[indices.cpu()]

    torch.testing.assert_close(out_cpu, ref)
    print("Test passed.")


if __name__ == "__main__":
    test_index_select()
