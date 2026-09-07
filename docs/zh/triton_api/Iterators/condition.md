# triton.language.condition

## 1. OP 概述

简介：`condition` 是 while 循环条件包装器，用于在 `@triton.jit` 函数中注解 while 循环的条件表达式，并允许用户向编译器传递额外的优化属性。

```python
triton.language.condition(arg1, disable_licm=False)
```

## 2. OP 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `arg1` | `tensor` | 必需 | 循环条件表达式，需为标量张量；GPU 上为标量布尔张量（`bool`/`int1`），Ascend A2/A3/950 额外支持 uint8、int8、int16、int32、int64、fp16、fp32、bf16 类型，如 `i < n` 的比较结果 |
| `disable_licm` | `bool` | `False` | 告知编译器不要将循环不变代码外提（LICM）到循环外。当循环内存在生命周期很长的变量时，禁用 LICM 可以避免产生过长的活动区间（live range） |

返回值：
无实际返回值，仅作为 `while` 语句的条件注解使用，条件本身会原样传递给循环控制。

### 2.2 类型支持

| 平台 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e(e4m3) | fp8e5(e5m2) | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU | × | × | × | × | × | × | × | × | × | × | × | × | × | × | √ |
| Ascend A2/A3 | √ | √ | × | √ | × | √ | × | √ | √ | √ | × | √ | × | × | √ |
| Ascend 950 | √ | √ | × | √ | × | √ | × | √ | √ | √ | × | √ | × | × | √ |

结论：GPU 仅支持 bool；Ascend A2/A3 与 Ascend 950 额外支持 uint8、int8、int16、int32、int64、fp16、fp32、bf16 类型。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

- `condition` 只能在 `@triton.jit` 函数内作为 `while` 语句的条件使用，在普通 Python 代码中调用无意义。
- 与 `tl.range` 的 `disable_licm` 参数一样，该功能在部分后端上属于编译器提示（hint），实际优化行为可能因后端而异。

### 2.4 使用方法

以下示例实现了带循环的 kernel，并对比 `disable_licm` 的效果：

```python
import triton
import triton.language as tl


@triton.jit
def while_default(n):
    i = 0
    while tl.condition(i < n):
        # 编译器默认会尝试将循环不变代码外提到循环外
        i = i + 1


@triton.jit
def while_no_licm(n):
    i = 0
    while tl.condition(i < n, disable_licm=True):
        # 禁用 LICM，避免循环内长生命周期变量产生过长的活动区间
        i = i + 1
```

编译后可通过 LLVM IR 验证 `disable_licm=True` 时生成了 `llvm.licm.disable` 标记，而默认情况下没有该标记。
