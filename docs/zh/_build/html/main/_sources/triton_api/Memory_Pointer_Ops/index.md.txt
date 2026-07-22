# 内存/指针操作

|api|简要说明|
|--|--|
|[load](./tl.load.md) |返回一个张量，其值从由指针定义的内存位置加载|
|[store](./tl.store.md) |将数据张量存储到由指针定义的内存位置|
|[make_block_ptr](./tl.make_block_ptr.md) |返回指向父张量中某个块的指针|
|[advance](./tl.advance.md) |推进一个块指针|
|[load_tensor_descriptor](./load_tensor_descriptor.md) | 从张量描述符加载数据块 |
|[make_tensor_descriptor](./make_tensor_descriptor.md) | 创建张量描述符对象 |

```{toctree}
:maxdepth: 3
:hidden:
tl.load.md
tl.store.md
tl.make_block_ptr.md
tl.advance.md
load_tensor_descriptor.md
make_tensor_descriptor.md
