# triton.autotune

```python
triton.autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None, warmup=25, rep=100, use_cuda_graph=False)
```

用于自动调优 triton.jit 函数的装饰器。

```python
@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
  ],
  key=['x_size'] # the two above configs will be evaluated anytime  上面两个配置会随时解析
                 # the value of x_size changes  变量 x_size 的值发生了变化
)
@triton.jit
def kernel(x_ptr, x_size, **META):
    BLOCK_SIZE = META['BLOCK_SIZE']
```

- 注：当所有配置都被解析时，内核将运行多次。也就是说内核更新的任何值都会进行多次更新。为了避免这种不希望出现的行为，可以使用 `reset_to_zero` 参数，该参数会在运行任何配置之前将提供的张量值重置为零。
- 注：如果环境变量 `TRITON_PRINT_AUTOTUNING` 设置为 `"1"`，Triton 会在每次自动调优内核后向标准输出 (stdout) 打印一条消息，包括自动调优所花费的时间和最佳配置。

**参数：**

- `configs (list[triton.Config])` - `triton.Config` 对象列表。
- `key (list[str])` - 参数名列表，当值发生改变时将触发对所有配置的解析。
- `prune_configs_by (dict)` - 修剪配置的函数字典。包含以下字段：
  - `'perf_model'`: 性能模型，用于预测不同配置的运行时间，返回运行时间
  - `'top_k'`: 要进行基准测试的配置数量
  - `'early_config_prune'`（可选）: 用于提前修剪配置的函数（例如，`num_stages`）。它接收 `configs: List[Config]` 作为输入，并返回修剪后的配置
- `reset_to_zero (list[str])` - 参数名称列表，将在任何配置解析之前被重置为零。
- `restore_value (list[str])` - 参数名称列表，这些参数的值将在解析任何配置之后恢复。
- `pre_hook (lambda args, reset_only)` - 一个将在调用内核之前被调用的函数。该参数会覆盖 `reset_to_zero` 和 `restore_value` 的默认 `pre_hook`。
  - `args`：传递给内核的参数列表
  - `reset_only`：一个布尔值，表示 `pre_hook` 是否仅用于重置值，而没有对应的 `post_hook`
- `post_hook (lambda args, exception)` - 一个将在调用内核之后被调用的函数。该参数会覆盖 `restore_value` 的默认 `post_hook`。
  - `args`：传递给内核的参数列表
  - `exception`：在出现编译或运行时错误情况下，由内核引发的异常
- `warmup (int)` - 传递给基准测试的预热时间（以毫秒为单位），默认值为 25。
- `rep (int)` - 传递给基准测试的重复时间（以毫秒为单位），默认值为 100。
- `use_cuda_graph (bool)` - 是否使用 CUDA Graph 进行性能测量（默认为 `False`）。
