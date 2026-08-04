# 原子操作

|api|简要说明|
|--|--|
|[atomic_add](./atomic_add.md)  |在由 pointer 指定的内存位置执行原子加法 |
|[atomic_and](./atomic_and.md)  |在由 pointer 指定的内存位置执行原子逻辑与操作 |
|[atomic_cas](./atomic_cas.md)  |在由 pointer 指定的内存位置执行 1 个原子比较并交换操作 |
|[atomic_max](./atomic_max.md)  |在由 pointer 指定的内存位置执行 1 个原子最大值操作 |
|[atomic_min](./atomic_min.md)  |在由 pointer 指定的内存位置执行 1 个原子最小值操作 |
|[atomic_or](./atomic_or.md)  |在由 pointer 指定的内存位置执行 1 个原子逻辑或操作 |
|[atomic_xchg](./atomic_xchg.md)  |在由 pointer 指定的内存位置执行 1 个原子交换操作 |
|[atomic_xor](./atomic_xor.md)  |在由 pointer 指定的内存位置执行原子逻辑异或操作 |

```{toctree}
:maxdepth: 3
:hidden:
atomic_add.md
atomic_and.md
atomic_cas.md
atomic_max.md
atomic_min.md
atomic_or.md
atomic_xchg.md
atomic_xor.md
```
