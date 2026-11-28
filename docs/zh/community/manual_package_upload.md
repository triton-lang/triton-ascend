# 手动上传 wheel 和 npuir 包

社区维护者可以通过 GitHub Actions 的 **Upload Wheels** 工作流将预构建包发布到 OBS。每次运行只上传一种包。上传 Triton-Ascend wheel 时提供下载直链，上传 npuir 时选择归档目录。

## 打开上传表单

1. 使用对社区仓库有写权限的 GitHub 账号打开 [Upload Wheels 工作流](https://github.com/triton-lang/triton-ascend/actions/workflows/upload-wheels.yml)。
2. 点击 **Run workflow**，选择包含目标工作流版本的分支。
3. 按下方对应包类型填写参数，核对后点击表单底部的 **Run workflow**。进入本次运行即可查看上传日志。另一个包类型对应的作业会显示为 **Skipped**。

## 上传 npuir

勾选 `upload_npuir`，将 `wheel_urls` 留空。流水线从 `obs://ascend-cann-open/Triton_Innersource/npu-ir/version/` 下选择归档，一次发布 aarch64 和 x86_64 两个架构。即使填了 wheel 链接，此模式也会忽略它。

| 表单字段 | 上传最新 daily 包 | 上传指定构建的示例 |
| --- | --- | --- |
| `npuir_archive` | 保留默认值 `npu-ir-latest` | `stable_20261116080055` |
| `npuir_version` | `2.0.0`（与归档文件名一致） | `2.0.0` |
| `npuir_tag` | 可留空 | `20261116080055` |
| `npuir_overwrite` | 默认不勾选 | 默认不勾选 |

`npuir_archive` 只填目录名。要复现某次构建，填写 `stable_` 加 14 位时间戳。`npuir_tag` 只影响发布文件名，可用来区分同一版本的多次构建。

打开 **Upload archived npuir to OBS** 作业，在最后一步的日志中核对两个文件名和各自的 SHA256。按上表填写指定构建后，目标目录 `obs://triton-ascend-artifacts/cann/` 下应有以下两个文件。

- `ascendnpu-ir_2.0.0_linux-aarch64-20261116080055.tar.gz`
- `ascendnpu-ir_2.0.0_linux-x86-20261116080055.tar.gz`（对应源归档的 `linux-x86_64`）

如需让集成测试使用这批包，将日志中的文件名和 SHA256 按架构更新到 `cmake/bisheng-version.txt`。

## 上传 wheel

1. 保持 `upload_npuir` 未勾选。在 `wheel_urls` 中填写至少一个可直接下载的 HTTPS 链接，链接路径以 `.whl` 结尾。多个链接用空格或换行分隔。
2. 在 `new_version` 中选择版本处理方式。默认值 `auto` 会将构建时间戳缩短为日期，例如 `3.6.0+dev20261027220530` 变为 `3.6.0+dev20261027`。清空字段可保留原版本，填写具体版本（如 `3.2.2.dev2`）会将本次所有 wheel 改为该版本。
3. 核对 `obs_dir`。默认上传到 `obs://ascend-artifcat-packages/pypi/packages/ascend/repos/pypi/nightly/triton-ascend/`。同名 wheel 会被覆盖，提交前请核对改写后的版本和目标目录。

确认 **Upload prebuilt wheels to OBS** 作业成功后，在目标目录的 `index.html` 中核对发布后的 wheel 文件名。使用默认目录时，可直接查看 [Triton-Ascend nightly 包列表](https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/triton-ascend/)。

## 常见问题

| 运行结果 | 处理方法 |
| --- | --- |
| 找不到 **Run workflow** 按钮 | 确认已登录有仓库写权限的账号，并已在 Actions 左侧选中 **Upload Wheels**。[GitHub 手动运行工作流说明](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow) |
| npuir 下载返回 404 | 核对 `npuir_archive` 和 `npuir_version`，确认该目录下同时有 `linux-aarch64` 和 `linux-x86_64` 的源包。 |
| npuir 提示 `already exists` | 任一架构有同名包时，默认在上传前停止。要保留两次构建，换一个 `npuir_tag`。确认要替换目标文件时，勾选 `npuir_overwrite` 后重新运行。 |
| wheel 提示 `version rewrite would collide` | 多个输入包改写后文件名相同。只保留要发布的构建，或分次运行并指定不同的 `new_version`。 |
| 提示 `OBS credentials are empty` | 请仓库管理员检查 Actions Secrets。wheel 使用 `NIGHTLY_OBS_AK` 和 `NIGHTLY_OBS_SK`，npuir 使用 `OBS_AK` 和 `OBS_SK`。 |
