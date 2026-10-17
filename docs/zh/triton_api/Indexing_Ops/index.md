# 索引与元素操作

|api|简要说明|
|--|--|
|[flip](./flip.md) |沿着维度 dim 翻转张量 x |
|[where](./where.md) |根据 condition 返回来自 x 或 y 的元素组成的张量 |
|[swizzle2d](./swizzle2d.md) |将行主序排列为 size_i * size_j 的矩阵的索引，转换为每组 size_g 行的列主序矩阵的索引 |
|[gather](./gather.md) | 对`src`tensor沿`axis`维度按照`index`执行gather操作 |

```{toctree}
:maxdepth: 3
:hidden:
flip.md
where.md
swizzle2d.md
gather.md
```
