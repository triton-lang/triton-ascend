# 编译器提示操作

|api|简要说明|
|--|--|
|[debug_barrier](./debug_barrier.md) |插入 1 个屏障以同步 1 个块中的所有线程 |
|[max_constancy](./max_constancy.md) |告知编译器 input 中的第 1 个值是常量 |
|[max_contiguous](./max_contiguous.md) |告知编译器 input 中的第 1 个值是连续的 |
|[multiple_of](./multiple_of.md) |告知编译器 input 中的所有值都是 value 的倍数 |
|[assume](./assume.md)         | 用于向编译器提供条件假设信息，允许编译器基于已知为真的条件进行优化。 |

```{toctree}
:maxdepth: 3
:hidden:
debug_barrier.md
max_constancy.md
max_contiguous.md
multiple_of.md
assume.md
```
