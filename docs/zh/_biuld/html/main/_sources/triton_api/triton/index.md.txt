# triton

| API | 简要说明 |
|-----|----------|
| [jit](./jit.md) | JIT 装饰器 - 使用 Triton 编译器编译函数 |
| [autotune](./autotune.md) | 用于自动调优一个经过 `triton.jit` 编译的函数的装饰器 |
| [heuristics](./heuristics.md) | 用于指定如何计算某些元参数值的装饰器 |
| [Config](./Config.md) | 一个表示自动调优器可以尝试的可能内核配置的对象 |

```{toctree}
:maxdepth: 3
:hidden:

jit.md
autotune.md
heuristics.md
Config.md
```
