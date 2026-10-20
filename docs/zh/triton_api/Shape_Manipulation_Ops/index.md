# Shape操作

|api|简要说明|
|--|--|
|[broadcast](./broadcast.md) | 尝试将两个给定的块广播到一个共同兼容的 shape |
|[broadcast_to](./broadcast_to.md) | 尝试将给定的张量广播到新的 shape |
|[expand_dims](./expand_dims.md) | 通过插入新的长度为 1 的维度来扩展张量的形状
|[interleave](./interleave.md) | 沿着最后一个维度交错两个张量的值 |
|[join](./join.md) | 在一个新的次要维度中连接给定的张量 |
|[permute](./permute.md) | 排列张量的维度 |
|[ravel](./ravel.md) | 返回 x 的连续扁平视图 |
|[reshape](./reshape.md) | 返回一个具有与输入相同元素数但具有提供的形状的张量|
|[split](./split.md) | 将张量沿其最后一个维度分成两部分，该维度大小必须为 2 |
|[trans](./trans.md) | 将张量转置 |
|[view](./view.md) | 返回具有与输入相同元素但形状不同的张量 |

```{toctree}
:maxdepth: 3
:hidden:
broadcast.md
broadcast_to.md
expand_dims.md
interleave.md
join.md
permute.md
ravel.md
reshape.md
split.md
trans.md
view.md
```
