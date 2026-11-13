# [RFC]: 扩展 A5 正向卷积与转置卷积接口

### 动机

A5 卷积需要统一的接口来表达非对称 padding、指定输出类型，以及对输入通道分片产生的卷积结果进行累加。如果每一片都返回低精度张量，就会在归约完成前损失精度。将累加写成独立的加法，也无法向后端直接表达前后两次卷积之间的累加关系。

本提案扩展 `al.conv1d` 和 `al.conv2d`，并新增独立的 `al.conv_transpose1d` 和 `al.conv_transpose2d` 接口。四个接口采用统一的 FP32 累加约定，通过 `out_dtype` 指定输出类型。调用方可以在第一片传入 bias，后续片持续累加 FP32 结果，最后再转换输出精度。

### 提议变更

#### 1. 功能范围与兼容性

| 方向 | 提议行为 |
| --- | --- |
| 正向卷积 | 为现有 Conv1D/2D 接口增加输出类型选择和 FP32 累加能力，保留对称、非对称、`same` 和 `valid` padding。 |
| 转置卷积 | 提供独立的 1D/2D 接口，支持 stride、dilation、groups、对称与非对称 padding，以及 `output_padding`。 |
| 逻辑布局 | 支持带 batch 和不带 batch 的通道优先布局；输出是否包含 batch 维与输入一致。 |
| 累加 | 使用满足约束的前一次卷积 FP32 结果作为 `acc`，包括通过普通 `for` 循环传递该结果。 |
| 首期目标 | 新增的累加和转置卷积路径面向 A5。 |

正向卷积和转置卷积分别使用独立的公开接口及算子定义。

Conv1D 的公开参数名为 `padding`，旧参数名 `padding_size` 不作为兼容别名保留，调用方需要迁移。输出默认行为也需要明确说明：这些接口默认返回 FP32；需要 FP16 或 BF16 结果时，应显式设置 `out_dtype`。

#### 2. 公开接口

原型版本提供以下接口签名：

```python
al.conv1d(
    input, weight, bias=None,
    stride=None, padding=None, dilation=None, groups=None,
    out_dtype=tl.float32, acc=None,
)

al.conv2d(
    input, weight, bias=None,
    stride=None, padding=None, dilation=None, groups=None,
    out_dtype=tl.float32, acc=None,
)

al.conv_transpose1d(
    input, weight, bias=None,
    stride=1, padding=0, output_padding=0, groups=1, dilation=1,
    out_dtype=tl.float32, acc=None,
)

al.conv_transpose2d(
    input, weight, bias=None,
    stride=1, padding=0, output_padding=0, groups=1, dilation=1,
    out_dtype=tl.float32, acc=None,
)
```

现有正向接口省略几何参数时，内部将其归一化为 `stride=1`、`padding=0`、`dilation=1`、`groups=1`，保留原有参数顺序和默认行为。转置卷积的参数顺序参考 [PyTorch 函数式接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.conv_transpose2d.html)，在末尾追加 `out_dtype` 和 `acc`。

`out_dtype` 必须是编译期确定的 Triton dtype。省略时使用 `tl.float32`。`acc=None` 表示本次调用开始一条新的卷积累加链。

#### 3. 形状与 padding

以下用 `S` 表示空间维度：1D 包含一维，2D 包含两维；`K` 表示对应的卷积核维度。

| 张量 | 正向卷积 | 转置卷积 |
| --- | --- | --- |
| 输入 | `(N, Cin, *S)` 或 `(Cin, *S)` | `(N, Cin, *S)` 或 `(Cin, *S)` |
| 权重 | `(Cout, Cin / groups, *K)` | `(Cin, Cout / groups, *K)` |
| Bias | `(Cout,)` | `(Cout,)` |
| 结果与 `acc` | `(N, Cout, *Sout)` 或 `(Cout, *Sout)` | `(N, Cout, *Sout)` 或 `(Cout, *Sout)` |

只有 batch 维可以省略。通道维和空间维必须显式保留，即使该维长度为 1。转置卷积的权重布局与正向卷积不同，需要按各自的布局进行校验。

几何参数在编译期确定。提议的有效性约束要求：张量各维尺寸、stride、dilation 和 groups 为正，padding 非负，通道分组关系一致，输出尺寸为正。原型版本的正向参数校验尚未完整覆盖这些要求。1D 的 stride 和 dilation 接受整数或单元素 tuple；2D 接受整数或 `(H, W)`。转置卷积的 `output_padding` 使用相同的逐轴归一化方式。

| 接口 | 数值形式的 `padding` | 字符串形式 |
| --- | --- | --- |
| `conv1d` | `p` 或 `(left, right)` | `'valid'`、`'same'` |
| `conv2d` | `p`、`(pad_h, pad_w)` 或 `(top, bottom, left, right)` | `'valid'`、`'same'` |
| `conv_transpose1d` | `p`、`(p,)` 或 `(left, right)` | 不支持 |
| `conv_transpose2d` | `p`、`(pad_h, pad_w)` 或 `(top, bottom, left, right)` | 不支持 |

整数表示所有空间轴均使用相同的对称 padding。2D 的二元 tuple 分别指定高、宽方向的对称 padding；四元 tuple 的顺序固定为**上、下、左、右**。本次原型中的正向 Conv1D 不接受单元素 padding tuple。

对于正向卷积，`valid` 表示不补零；`same` 仅支持 stride 为 1，与 [PyTorch Conv1D 的约定](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.conv1d.html)一致。每个空间轴的总 padding 为 `dilation * (K - 1)`，前侧取较小的一半，剩余部分放在后侧。因此，奇数和偶数卷积核都能保持输入空间尺寸。计算前需要将形状值从 `tl.constexpr` 中解包，确保传给 builder 的 padding 为普通整数。

正向卷积每个空间轴的目标输出尺寸公式为：

```text
O = floor((I + pad_before + pad_after - dilation * (K - 1) - 1) / stride) + 1
```

转置卷积每个空间轴的输出尺寸为：

```text
O = (I - 1) * stride - pad_before - pad_after
    + dilation * (K - 1) + output_padding + 1
```

`output_padding` 必须非负，并且在每个空间轴上小于 stride **或** dilation。它用于确定输出尺寸，不表示显式执行一次补零操作。非对称转置 padding 在对称接口的基础上，允许独立指定两侧的边界偏移。

#### 4. 输出类型与数值行为

A5 卷积路径保留 input、weight 和可选 bias 的原始元素类型。核心输入类型为 FP16、BF16 和 FP32，各操作数类型一致。无论请求哪种输出类型，HFusion 卷积的累加器和结果都使用 FP32。

| `out_dtype` | 结果处理 | 原型验证状态 |
| --- | --- | --- |
| 省略或 `tl.float32` | 直接返回 FP32 卷积结果。 | 已有 A5 数值测试覆盖。 |
| `tl.float16` 或 `tl.bfloat16` | 在卷积完成后转换 FP32 结果。 | 已有 A5 数值测试覆盖。 |
| `tl.float8e4nv`、`tl.float8e5`、`tl.float8e4b15` | 使用对应的输出转换路径。 | 前端与 IR 支持，尚无 A5 数值验收结论。 |
| FP64、非浮点类型、`tl.float8e4b8`、`tl.float8e5b16` | 明确报错。 | 不支持。 |

FP16 和 BF16 输出在 FP32 卷积结果之后显式生成 `arith.truncf`。FP8 路径属于实验性支持，其中 `float8e4b15` 使用专用编码转换。

#### 5. FP32 累加约定

不传 `acc` 时，本次调用计算新的卷积结果，并可选择加入 bias。传入 `acc` 时，本次调用将新的卷积贡献累加到之前的 FP32 结果中。Bias 只在开始累加链时传入。

前端检查 `acc` 必须为 `tl.float32` 张量，形状和轴顺序必须与当前卷积的逻辑输出完全一致，并且不能与 bias 同时传入。不支持广播；在非 A5 目标上传入 `acc` 会被拒绝。

后端负责校验累加链：前序结果必须来自同一种卷积算子，groups 相同，结果类型和形状完全一致，且累加器满足单一使用限制。一条累加链不能混用正向和转置卷积，也不能混用 1D 和 2D。零张量、任意 GM 加载结果、算术结果或 cast 结果都不能作为初始累加器。累加链必须从实际卷积结果开始，可以通过 `scf.for` 的循环携带值继续传递。

例如，在 Triton kernel 内，`groups=1` 的两个通道分片可以按以下方式累加。两次调用的逻辑输出形状相同，`x0` 与 `w0` 对应第一段输入通道，`x1` 与 `w1` 对应第二段输入通道：

```python
partial = al.conv1d(
    x0, w0, bias=bias, padding=(1, 2),
    out_dtype=tl.float32,
)
result = al.conv1d(
    x1, w1, padding=(1, 2),
    acc=partial, out_dtype=tl.bfloat16,
)
```

其他三个接口采用相同的累加方式，并遵循各自的权重布局。普通 `for` 循环中的携带值保持 FP32，直接作为下一次卷积的 `acc`。最后一次调用可以选择 FP16 或 BF16 输出，包括循环结束后继续计算的尾片。前端不应在累加链中主动插入中间 store、load 或低精度转换。

#### 6. IR 降级与职责划分

每个公开接口对应独立的 HFusion 算子：

| 公开接口 | HFusion 算子 |
| --- | --- |
| `al.conv1d` | `hfusion.conv1d` |
| `al.conv2d` | `hfusion.conv2d` |
| `al.conv_transpose1d` | `hfusion.conv_transpose1d` |
| `al.conv_transpose2d` | `hfusion.conv_transpose2d` |

这些算子继续使用现有的 `init` 操作数，并配合 `i1` 类型的 `init_condition`：

| 调用方式 | `init` | `init_condition` | HFusion 结果 |
| --- | --- | --- | --- |
| 不传 `acc` | 与逻辑输出形状一致的 FP32 `tensor.empty` | `true` | FP32 |
| 传入 `acc` | 调用方提供的 FP32 累加器 | `false` | FP32 |

`init_condition=true` 时，由后端初始化卷积累加器；`tensor.empty` 中未初始化的内容不参与累加，前端无需预先清零。为 `false` 时，直接使用之前的卷积结果，不额外生成加法算子。

```text
input / weight / 可选 bias，保持原始类型
                  |
                  v
     HFusion 卷积，init/result 为 FP32
                  |
         根据 out_dtype 转换输出
                  |
                  v
                结果
```

前端负责 API 参数归一化、形状与类型校验、C++ builder 和 TritonAscend IR 支持，以及上述降级过程。NPUIR/HFusion 负责累加器来源校验、卷积归一化、物理布局、HIVM 降级、模板选择和 L0C 复用。公开接口不要求调用方构造后端布局，也不要求通过正向卷积的 `dx` 模式表达转置卷积。

TA 的 IR 生成端与包内 NPUIR 读取端、编译器必须使用匹配的算子定义，包括 `init_condition` 和输入、结果使用不同元素类型的支持。仅替换 Python 文件不足以交付这项能力。

#### 7. 上游接入顺序

原型版本已实现上述路径。接入上游时，建议保留以下独立评审边界：

1. Conv1D 非对称 padding 及参数名迁移与其他功能分别评审。
2. 在匹配 HFusion 定义的基础上，接入正向 `out_dtype` 和 `acc`，包含普通循环及循环结束后尾片的回归测试。
3. 接入独立的转置卷积算子及对称、非对称 padding 行为。
4. 单独更新 API 文档和设备限制，并为每项功能提供对应的前端、IR 和 A5 验证。

开发变更以 `main-dev` 为目标分支。功能交付需要匹配的后端依赖和设备验证，不能仅因 Python 接口存在就宣称可用。

#### 8. 验证结果与验收标准

**388 项前端与 TTIR 测试通过**、**21 项 Conv1D padding 回归通过**、**12 组原生 IR 回归通过**，以及 **41 次 A5 数值测试执行成功**。设备验证目标为 `Ascend950PR_9579`，编译模式为 `simd_simt_template`，覆盖了代表性的正向和转置卷积、非对称 padding、FP32 累加、循环传递及最终输出转换。

41 次成功执行中，包含 7 个在调用处显式绑定 `tl.constexpr` 后通过的 2D 用例。这 7 个用例在原脚本中因通用嵌套 JIT tuple 参数处理失败，尚未进入卷积降级阶段；本次包没有修复该 JIT 问题。完整的 1,216 项外部用例矩阵仅完成收集，没有全部执行。这些结果验证了上述代表性场景的正确性。

上游验收应分别覆盖以下层次：

| 层次 | 验收内容 |
| --- | --- |
| Python 前端 | 带 batch 和不带 batch 的形状；全部约定的 padding 形式；奇偶卷积核的 `same`；非法几何参数与类型；bias/acc 互斥；acc 形状精确匹配；A5 设备限制。 |
| IR 与降级 | padding 归一化数值及属性类型正确；FP32 init/result；`init_condition` 正确；不额外添加累加加法或预清零；输出转换位于卷积之后；bytecode 生成端与读取端匹配。 |
| 控制流 | 直接累加、普通循环携带累加值、循环后继续调用卷积；拒绝非法初值和混合算子的累加链。 |
| A5 数值执行 | 使用能区分 padding 左右颠倒和输出错位的输入；与显式补零或裁剪的参考结果比较；覆盖通道分片、bias 仅加入一次、groups、支持的内存视图，以及最终 FP16/BF16/FP32 输出。 |
