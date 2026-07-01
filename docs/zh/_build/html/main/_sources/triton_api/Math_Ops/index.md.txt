# 数学操作

|api|简要说明|
|--|--|
|[add](./add.md) | 四则运算加法 ‘+’ |
|[sub](./sub.md) | 四则运算减法 ‘-’ |
|[mul](./mul.md) | 四则运算乘法 ‘*’ |
|[div](./div.md) | 四则运算除法 ‘/’ |
|[abs](./abs.md) |计算 x 的逐元素绝对值 |
|[neg](./neg.md) | 将tensor的值取负。 |   x
|[cdiv](./cdiv.md) |计算 x 除以 div 的向上取整除法 |
|[ceil](./ceil.md) |计算 x 的逐元素向上取整值 |
|[clamp](./clamp.md) |将输入张量 x 的值限制在 [min, max] 范围内 |
|[cos](./cos.md) |计算 x 的逐元素余弦值 |
|[div_rn](./div_rn.md) |计算 x 和 y 的逐元素精确除法（根据 IEEE 标准四舍五入到最近的值） |
|[erf](./erf.md) |计算 x 的逐元素误差函数 |
|[exp](./exp.md) |计算 x 的逐元素指数 |
|[exp2](./exp2.md) |计算 x 的逐元素指数（以 2 为底）|
|[fdiv](./fdiv.md) |计算 x 和 y 的逐元素快速除法 |
|[floor](./floor.md) |计算 x 的逐元素向下取整 |
|[fma](./fma.md) |计算 x、y 和 z 的逐元素融合乘加运算 |
|[log](./log.md) |计算 x 的逐元素自然对数 |
|[log2](./log2.md) |计算 x 的逐元素对数（以 2 为底）|
|[mod](./mod.md) | 取模运算 |
|[maximum](./maximum.md) |计算 x 和 y 的逐元素最大值 |
|[minimum](./minimum.md) |计算 x 和 y 的逐元素最小值 |
|[rsqrt](./rsqrt.md) |计算 x 的逐元素的平方根倒数 |
|[sigmoid](./sigmoid.md) |计算 x 的逐元素 sigmoid 函数值 |
|[sin](./sin.md) |Computes the element-wise sine of x. 计算 x 的逐元素正弦值 |
|[softmax](./softmax.md) |计算 x 的逐元素 softmax 值 |
|[sqrt](./sqrt.md) |计算 x 的逐元素快速平方根 |
|[sqrt_rn](./sqrt_rn.md) |计算 x 的逐元素精确平方根（根据 IEEE 标准四舍五入到最近的值） |
|[umulhi](./umulhi.md)  |计算 x 和 y 的 2N 位乘积的逐元素最高有效 N 位 |

```{toctree}
:maxdepth: 3
:hidden:
add.md
sub.md
mul.md
div.md
abs.md
neg.md
cdiv.md
ceil.md
clamp.md
cos.md
div_rn.md
erf.md
exp.md
exp2.md
fdiv.md
floor.md
fma.md
log.md
log2.md
mod.md
maximum.md
minimum.md
rsqrt.md
sigmoid.md
sin.md
softmax.md
sqrt.md
sqrt_rn.md
umulhi.md
