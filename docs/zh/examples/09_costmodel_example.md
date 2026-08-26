# Costmodel 端到端示例

本示例展示 costmodel 后端的基本调用流程：

- 使用 Triton 前端算子生成 TTIR；
- 为多个候选 config 构造 `costmodel_bench` 输入；
- 调用 `costmodel_bench` 得到每个 config 的预测耗时。

这个流程适合在 autotune 前快速筛掉预计性能较差的 config。示例只使用向量加法 kernel，便于聚焦 costmodel 的输入和返回值。

## 完整示例

完整代码见：{download}`09_costmodel_example.py <full_examples/09_costmodel_example.py>`

## 示例输出

不同版本的 costmodel 参数可能会使具体数值略有不同，但输出结构类似：

```text
block256: 0.098 us
block1024: 0.110 us
block2048: 0.126 us
```

`costmodel_bench` 的返回值是一个字典，key 为传入的 `config`，value 为预测耗时，单位是微秒。上层 autotune 逻辑可以按 value 排序，优先保留预测更快的 config。

## 关键点说明

1. `ASTSource + ast_to_ttir` 只生成 TTIR，不会真实编译或启动 kernel。
2. `config` 会影响 `tl.constexpr`，例如 `BLOCK_SIZE`，因此每个候选 config 都需要生成各自的 TTIR。
3. `costmodel_bench` 接收的每个元素至少包含 `config` 和 `ttir`，也可以附带 `arg_bindings`。
4. `arg_bindings` 用于把运行时整数参数绑定到 TTIR 中的 `%argN`。例如本例中 `n_elements=98432` 对应 `arg3=98432`。
5. 如果 kernel 中使用 `tl.program_id(0)`，通常需要传入 `pid_x=0`。如果还使用 `tl.num_programs(0)`，可额外传入 `num_programs_x=...`。
