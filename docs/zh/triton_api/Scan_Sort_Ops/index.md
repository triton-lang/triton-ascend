# 扫描/排序操作

|api|简要说明|
|--|--|
|[associative_scan](./associative_scan.md) |沿指定 axis 将 combine_fn 应用于 input 张量的每个元素和携带的值，并更新携带的值 |
|[cumprod](./cumprod.md) |返回沿指定 axis 的 input 张量中所有元素的累积乘积 |
|[cumsum](./cumsum.md)  |返回沿指定 axis 的 input 张量中所有元素的累积和 |
|[histogram](./histogram.md) |基于 input 张量计算 1 个具有 num_bins 个 bin 的直方图，每个 bin 宽度为 1，起始于 0 |
|[sort](./sort.md) |沿着指定维度对张量进行排序 |

```{toctree}
:maxdepth: 3
:hidden:
associative_scan.md
cumprod.md
cumsum.md
histogram.md
sort.md
