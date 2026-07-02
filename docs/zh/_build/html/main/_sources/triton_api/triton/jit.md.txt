# triton.jit

```python
triton.jit(fn: T)→ JITFunction[T]
triton.jit(*, version=None, repr: Callable | None = None, launch_metadata: Callable | None = None, do_not_specialize: Iterable[int] | None = None, debug: bool | None = None, noinline: bool | None = None)→ Callable[[T], JITFunction[T]]
```

使用 Triton 编译器的 JIT 编译函数的装饰器。

- 注:当调用 JIT 编译的函数时，如果参数具有 `.data_ptr()` 方法和 `.dtype` 属性，则会隐式转换为指针。

- 注:此函数将在 GPU 上编译和运行。它只能访问以下内容：
    - Python 原语，
    - Triton 包内的内置函数，
    - 此函数的参数，
    - 其他 JIT 编译的函数。

**参数：**`fn (Callable)` - 要进行 JIT 编译的函数
