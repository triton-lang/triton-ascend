# Ascend 社区用例完整验证

本入口运行仓库 python/test/unit/ 中的全部 458 个测试定义，包括 440 个顶层函数和
test_mxfp.py 中的 18 个 CPU 类方法。范围涉及 49 个测试文件，其中 test_module.py
是辅助模块。通过、失败、跳过和超时的用例都保留在清单中，每个定义的参数由当前环境下
的 pytest 完整收集。

用例源码继续放在原目录。仅为两个模块增加四处非 CUDA 收集保护，避免导入时直接查询
CUDA 能力；内核、参考值、参数、断言和原有跳过条件的目标要求保持原样。
这是供测试人员主动执行的完整验证入口，未接入日常 CI，也不按历史通过结果筛选用例。

## 执行

使用已经安装好 Triton-Ascend、Torch-NPU、CANN 和测试依赖的 Python 环境。
从仓库根目录执行以下命令，设备编号选择自己已确认空闲的物理 NPU。

~~~bash
# 只核对源码和 458 项清单，不导入 NPU 软件、不执行测试。
python third_party/ascend/tools/community/run.py --check

# 收集全部参数，保存各文件的收集日志和节点。
python third_party/ascend/tools/community/run.py \
  --npu-device 0 --output /path/to/community-run --collect-only

# 使用刚才的收集结果，执行全部参数。
python third_party/ascend/tools/community/run.py \
  --npu-device 0 --output /path/to/community-run --resume
~~~

也可以用一个新的 --output 目录直接运行，不加 --collect-only。
正常定义每批最多 32 个参数，涉及断言、自旋锁、子进程等的 31 个定义逐参数使用独立进程。
所有文件分别收集，兼顾同名模块和同目录裸导入。CPU 类方法保留自己的 CPU fixture。

普通失败会记录后继续。单个节点默认限时 180 秒；超时、异常退出或执行中断会停止调度，
保存已完成结果、最后启动的节点和剩余清单。CAS 用例曾出现设备等待超时，遇到这类情况后
应先检查设备健康，再使用相同命令加 --resume 继续。已完成节点不会被自动重试；
重新验证失败节点时使用新的输出目录。

~~~bash
# 验证一个定义的全部参数；也可传入收集结果中的完整参数 nodeid。
python third_party/ascend/tools/community/run.py \
  --npu-device 0 --output /path/to/community-float-annotation \
  --select python/test/unit/language/test_annotations.py::test_float_annotation
~~~

--select 可以重复使用，也可以指定完整源文件。需要调整限时或普通批次大小时，使用
--node-timeout 和 --batch-size。更换测试源码、安装的软件或编译选项后，使用新输出目录。
新入口使用当前已安装的测试软件和编译设置，不安装依赖、不改编译选项，不重写库的默认缓存目录。
常规编译缓存与临时文件放在本次输出目录；测试自身的缓存与配置重置操作仍按原源码执行。

## 查看结果

- nodes.json：本次选择范围内的完整参数节点。
- progress.json、remaining.txt：各节点状态、累计数量、收集错误、批次异常和剩余节点。
- collection/：各文件的收集结果、节点及参数取值。
- batches/：每批实际命令、完整输出、JUnit、分阶段事件、退出码和超时记录。

节点状态区分 passed、failed、skipped、xfailed、xpassed、error、timeout 和 crash；
还没执行的节点保留在 remaining。结果存在失败、异常、超时、意外通过或未完成时，
入口返回非零退出码。收集不完整时保留错误并停止执行，避免把缺失文件误报为全量完成。

注册 Torch-NPU 后会恢复调用者原有的 TORCH_DEVICE_BACKEND_AUTOLOAD 值，保证测试子进程
继承原设置。观察插件只记录 pytest 结果，不替换 CUDA 接口，也不增加 skip/xfail 标记。

## A5 历史对照

a5_reference_results.csv 保存此前 A5 使用 9 月 10 日编译软件完成的首次正式结果，
按完整源路径、类名和函数名对应这 458 个定义。它是历史记录，不是本 PR 新执行的结果，
也不决定新运行中的跳过或预期失败。

| 范围 | 参数节点 | 通过 | 失败 | 跳过 | 预期失败 | 超时 |
|---|---:|---:|---:|---:|---:|---:|
| original-default：原源码可收集的定义 | 13,429 | 7,766 | 3,556 | 2,096 | 1 | 10 |
| adapted-default：两个模块完成收集保护后的定义 | 5,493 | 125 | 2,405 | 2,963 | 0 | 0 |
| 合计 | 18,922 | 7,891 | 5,961 | 5,059 | 1 | 10 |

18 个 CPU 类方法对应的 24 个历史参数全部通过。追加诊断和候选修复验证均未加入以上统计。
软件修复、目标设备及准备方式会影响新结果，应按新日志判断；例如历史打印/AOT 加载失败
和默认目录断言曾受旧执行插件影响，新入口没有沿用这些插件行为。

## 维护入口

manifest.json 登记全部定义与进程隔离要求。上游新增、删除或重命名测试时，
--check 会报告清单差异，需要同步更新清单。历史 CSV 保持原记录。
运行工具自身的 CPU 检查使用：

~~~bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  third_party/ascend/tools/community/tests
~~~
