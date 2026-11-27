# TLA 类型与参数组合校验

TLA 在创建 IR 之前检查接口支持的类型与参数组合。对于不支持的调用，错误指出接口、类型组合、请求参数和允许的值。例如：

```text
tla.simd.vcvt f32 -> i32 does not support round=ROUND.O; supported round: ROUND.R, ROUND.A, ROUND.F, ROUND.C, ROUND.Z
```

`ROUND.O` 是 Von Neumann round-to-odd；它的存在不表示每一种转换都支持这个模式。当前 `vcvt` 的 `f32 -> int32` 和 `f16 -> int16` 只允许 R/A/F/C/Z。此检查不会禁止 `f32 -> f16` 的 O 模式，也不会把 O 自动替换成其他舍入方式。

## 公共检查器

`language/tla/_type_contract.py` 提供纯 Python 的 `TypeContract`、`ParameterRule` 和 `check_type_contract`。其他 TLA 接口也可以使用，不依赖 Triton、Torch、NPU 驱动或当前活动设备。

接口把参数规范化后，以明确的顺序传入类型签名。例如转换使用 `(源类型, 目标类型)`，相同类型的逐元素操作使用 `(元素类型,)`。整数有符号性和浮点格式必须保留：`i32` / `u32`、`f16` / `bf16` 是不同类型。

下面展示限制类型与舍入模式的写法，作为接口内部的设计示例：

```python
from ._type_contract import ParameterRule, TypeContract, check_type_contract

contract = TypeContract(
    signatures=(("f32", "i32"), ("f32", "f16")),
    parameters=(
        ParameterRule(
            ("f32", "i32"), "round",
            ("ROUND.R", "ROUND.A", "ROUND.F", "ROUND.C", "ROUND.Z"),
        ),
    ),
)

check_type_contract(
    "tla.simd.vcvt", (src_type, dst_type), contract,
    options={"round": round_name}, target=compile_target,
)
```

- `signatures` 指定完整的允许类型集合。不匹配则拒绝；空集合表示没有支持的签名。
- `signatures=None` 表示保留接口原有的类型检查，只增加已确认的组合限制。它不证明其他组合已经获得后端支持。
- `parameters` 按完整类型签名匹配规则。同一签名的多项规则都必须满足，所需参数不能省略；可用于 `round`、`mode`、`dist` 等规范化参数。
- `target` 用于错误中的目标说明。不同目标的能力有差异时，调用方应从编译选项选择对应规则，不应读取当前设备来猜测目标；检查器不会根据这个字符串自动选择硬件能力。
- 相同类型关系、mask 形状、地址对齐、动态值等检查仍由接口或已有公共 helper 处理。

## 当前接入范围

`simd_semantic.py` 集中保存 SIMD 规则。原有的逐元素 `_OP_ELEM_SUPPORT` 已迁入并经公共检查器执行，允许的类型集合保持原有约定；诊断的支持列表直接来自这些集合。`vcvt` 增加 f32→i32、f16→i16 的舍入组合限制，检查发生在构造默认 mask 和转换 IR 之前。

转换规则依据 CANN 9.3 B066 的 `__clang_cce_vector_intrinsics_legacy.h` 中 `__VF_VCVTFI_SAT` 的公开参数检查；AVE 将这两个转换的舍入值直接传入 intrinsic。首批仅限制这两个已确认的类型组合，不把它扩展成所有类型或芯片的完整能力表。若后续更换实现路径或新增目标能力，应同时更新规则和测试。

新增规则应记录对应的后端实现或文档依据，并覆盖：非法输入在创建 IR 前被拒绝、合法输入继续到达 builder、符号性/格式区分，以及相邻合法组合没有被误拒绝。需要证明计算语义时，再增加目标设备上的数值测试。AVE 自身的 verifier 仍需独立保证 IR 合法性。
