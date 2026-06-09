# triton 总览

## triton op 支持度总览

|                          |        Triton Op       | int8 | int16 | int32 | uint32 | int64 | fp16 | fp32 | bf16 | bool |
|:------------------------:|:----------------------:|------|-------|-------|--------|-------|------|------|------|------|
|       Creation Ops       | arange                 | ×    | ×     | ✓     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | cat                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | full                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | zeros                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | zeros_like             | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | cast                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|  Shape Manipulation Ops  | broadcast              | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | broadcast_to           | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | expand_dims            | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | interleave             | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | join                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | permute                | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ✓    |
|                          | ravel                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | reshape                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | split                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | trans                  | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ✓    |
|                          | view                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|    Linear Algebra Ops    | dot                    | ✓    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | dot_scaled             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|    Memory/Pointer Ops    | load                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | store                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | make_block_ptr         | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | make_tensor_descriptor | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | load_tensor_descriptor | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | store_tensor_descriptor| ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | advance                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|       Indexing Ops       | flip                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | where                  | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ✓*   |
|                          | swizzle2d              | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|         Math Ops         | add                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | sub                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | mul                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | div                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | floordiv(//)           | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓*   |
|                          | mod                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | neg                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | invert(~)              | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | and(&)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | or(\|)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | xor(^)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | not(!)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | lshift(<<)             | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | rshift(>>)             | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | gt                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | ge                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | lt                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | le                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | eq                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | ne                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | logical and            | ✓    | ✓     | ✓     | ✓      | ✓     | ✓    | ✓   | ✓    | ✓    |
|                          | logical or             | ✓    | ✓     | ✓     | ✓      | ✓     | ✓    | ✓   | ✓    | ✓    |
|                          | abs                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | cdiv                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | ceil                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | clamp                  | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | cos                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | div_rn                 | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | erf                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | exp                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | exp2                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | fdiv                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | floor                  | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | fma                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | log                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | log2                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | maximum                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | minimum                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | round                  | ×    | ×     | ×     | ×      | ×     | ×    | ✓    | ×    | ×    |
|                          | rsqrt                  | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sigmoid                | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sin                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | softmax                | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sqrt                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sqrt_rn                | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | umulhi                 | ×    | ×     | ✓     | ×      | ×     | ×    | ×    | ×    | ×    |
|       Reduction Ops      | argmax                 | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | argmin                 | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | max                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | min                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*    |
|                          | reduce                 | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*    |
|                          | sum                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*    |
|                          | xor_sum                | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓*    |
|       Scan/Sort Ops      | associative_scan       | ✓    | ✓     | ✓     | ✓      | ✓     | ✓    | ✓    | ×    | ✓    |
|                          | cumprod                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | cumsum                 | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | histogram              | ×    | ×     | ✓      | ✓       | ✓      | ×    | ×    | ×    | ×    |
|                          | sort                   | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | gather                 | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|        Atomic Ops        | atomic_add             | ✓    | ✓     | ✓     | ✓       | ×     | ✓    | ✓    | ✓    | ×    |
|                          | atomic_and             | ✓    | ✓     | ✓     | ✓       | ✓     | ×    | ×    | ×    | ×    |
|                          | atomic_cas             | ×    | ✓     | ✓     | ✓       | ✓     | ✓    | ✓    | ×    | ×    |
|                          | atomic_max             | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | atomic_min             | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | atomic_or              | ✓    | ✓     | ✓     | ✓       | ✓     | ×    | ×    | ×    | ×    |
|                          | atomic_xchg            | ✓    | ✓     | ✓     | ✓       | ✓     | ✓    | ✓    | ×    | ×    |
|                          | atomic_xor             | ✓    | ✓     | ✓     | ✓       | ✓     | ×    | ×    | ×    | ×    |
| Random Number Generation | randint4x              | ✓     | ✓      | ✓     | ✓       | ×     | ×    | ×    | ×    | ✓     |
|                          | randint                | ✓     | ✓      | ✓     | ✓       | ×     | ×    | ×    | ×    | ✓     |
|                          | rand                   | ×    | ×     | ×     | ×      | ×     | ✓     | ✓    | ✓     | ✓     |
|                          | randn                  | ×    | ×     | ×     | ×      | ×     | ✓     | ✓    | ✓     | ✓     |
|         Iterators        | range                  | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | static_range           | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|      Inline Assembly     | inline_asm_elementwise | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|     Compiler Hint Ops    | assume                 | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ✓    |
|                          | debug_barrier          | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | max_constancy          | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | max_contiguous         | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | multiple_of            | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|         Debug Ops        | static_print           | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | static_assert          | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | device_print           | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ×    | ✓    |
|                          | device_assert          | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ✓    |

## 约束说明

- dot: 两个输入A[batch(optional), M, K], B[batch(optional), K, N]。

- gather: triton.gather(x, index, axis)，假设x的shape为n维度，目前只支持axis=n-1。

- permute: triton.permute(x, dims)，不支持dims=[2, 1, 0]。

- trans: triton.trans(x, dims)，不支持dims=[2, 1 , 0]。

- device_print: 需要增加1个环境变量，TRITON_DEVICE_PRINT=1。

- device_assert: 生效需要设置两个环境变量，TRITON_DEBUG=1，TRITON_DEVICE_PRINT=1。

- atomic_add: 昇腾不支持atomic_add实现多核add+保存中间结果，需要修改成普通add来保存中间结果

- atomic类op: 对于昇腾后端，sem只支持默认值"acq_rel"模式，其他值均按默认值处理；scope只支持默认值"gpu"，其他值均按默认值处理

- atomic_or/atomic_xor/atomic_and/atomic_xchg/atomic_cas: 昇腾暂不支持在loop中使用

- permute: 不支持不相邻轴转置，如`(0, 1, 2) -> (2, 1, 0)`

- trans: 不支持不相邻轴转置，如`(0, 1, 2) -> (2, 1, 0)`

- umulhi: 不支持负数输入

- mod: int64仅支持处理 -2^24 ~ 2^24 范围内的数值

- rand类op: 所支持的数据类型仅针对算子的输出。

- tensor_descriptor类op: 当前仅支持绑定使用，即 make/load/store_tensor_descriptor 需配套使用

- ALL: int8类型由于特殊处理，会占用更大的片上空间，编译时容易造成ub overflow报错，通常调整tiling即可解决
- ALL: triton kernel中同时存在所有tensor总和不能超过96KB，若关闭double buffer，则不能超过192KB
- ALL: 所有tensor不允许某个shape的size小于1
- ALL: ✓*表示triton内部将bool类型转为int8类型进行运算，并能够执行得到结果的OP
- ALL: 不支持使用shape为"[[]]"的标量tensor进行计算
