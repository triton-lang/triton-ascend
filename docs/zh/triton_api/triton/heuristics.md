# triton.heuristics

```python
triton.heuristics(values)
```
用于指定如何计算某些元参数值的装饰器。这在自动调优成本过高或不适用的情况下非常有用。

```python
@triton.heuristics(values={'BLOCK_SIZE': lambda args: 2 ** int(math.ceil(math.log2(args[1])))})
@triton.jit
def kernel(x_ptr, x_size, **META):
    BLOCK_SIZE = META['BLOCK_SIZE'] # smallest power-of-two >= x_size  最小的 2 的幂 >= x_size
```

**参数：** `values (dict[str, Callable[[list[Any]], Any]]**)` - 包含元参数名称和计算元参数值的函数的字典。每个这样的函数都接受一个位置参数列表作为输入。
