# triton.Config

```python
classtriton.Config(self, kwargs, num_warps=4, num_stages=2, num_ctas=1, maxnreg=None, pre_hook=None)
```
表示自动调优可能尝试的内核配置的对象

**变量：**
- kwargs – 1 个元参数字典，用于作为关键字参数传递给内核。

- num_warps – 在为 GPU 编译时内核使用的线程数。例如，如果 num_warps=8，则每个内核实例将自动并行化，使用 8 * 32 = 256 个线程协作执行。

- num_stages – 编译器在软件流水线循环时应使用的阶段数。对于 SM80+ GPU 上的矩阵乘法工作负载非常有用。

- num_ctas - 块集群中的块数。仅适用于 SM90+。

- maxnreg - 单个线程可以使用的最大寄存器数。对应于 ptx 的 .maxnreg 指令。并非所有平台都支持。

- pre_hook – 在调用内核之前将被调用的函数。该函数的参数是 args。

```python
__init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1, maxnreg=None, pre_hook=None)
```

**方法：**
| init(self, kwargs[, num_warps, ...]) |
|-----|
|all_kwargs (self)|
