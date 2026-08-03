# DynamicCVPipeline 门禁设计文档

## 1. 触发条件

- **自动触发**：PR 修改以下目录时
  - `third_party/ascend/lib/DynamicCVPipeline/**`
  - `third_party/ascend/include/DynamicCVPipeline/**`
- **手动触发**：`workflow_dispatch`，可在 Actions 页面手动启动

## 2. 整体流程

```
PR 提交/更新 (匹配路径)
        │
        ▼
┌──────────────────────────────────────────────┐
│  Job: remote-test (ubuntu-latest)            │
│                                              │
│  1. 安装 sshpass                             │
│  2. SSH 到远端服务器 (61.47.16.82)            │
│     sshpass + 用户 z00896713                 │
│                                              │
│  3. 远端执行:                                 │
│     docker start z00896713                   │
│     docker exec z00896713 bash -c '...'      │
│       chmod -R a+rwX /home/z00896713         │
│       cd /home/z00896713/ci                  │
│       python ci.py                           │
│                                              │
│  4. docker cp hello.txt 到远端临时目录        │
│  5. scp hello.txt 回到 GitHub Actions runner │
│  6. 上传为 PR artifact                       │
│     artifact 名: hello-txt-pr-<PR_NUMBER>    │
│  7. 清理远端临时文件                          │
└──────────────────────────────────────────────┘
```

## 3. 远端服务器

| 项 | 值 |
|---|---|
| IP | 61.47.16.82 |
| 用户 | z00896713 |
| 环境 | Docker 容器 `z00896713` |
| 工作目录 | `/home/z00896713/ci` |
| 入口脚本 | `python ci.py` |
| 产物 | `hello.txt` |

密码通过 GitHub Secret `DYNAMIC_CV_TEST_PASSWORD` 注入，**绝不硬编码在 workflow 文件中**。

## 4. ci.py 说明

`ci.py` 是远端 Docker 容器内的测试入口脚本，位于 `/home/z00896713/ci/ci.py`。

- **功能**：拉取 PR 代码，分别对 base 和 head 运行 DynamicCVPipeline 相关测试，生成对比结果
- **输入**：通过环境变量或参数获取 PR 信息（PR number、base commit、head commit）
- **输出**：`/home/z00896713/ci/hello.txt`

## 5. 产物查看

每个 PR 运行后会生成一个 artifact，名称为 `hello-txt-pr-<PR_NUMBER>`。查看方式：

1. 进入 PR 页面
2. 点击 CI 完成的 workflow run
3. 在 Summary 页面底部下载 artifact

如果同一个 PR 多次触发，后来的 run 会覆盖之前的 artifact（同名）。

## 6. 并发策略

多个 PR 同时触发时**排队串行执行**（`cancel-in-progress: false`），不会因为新 PR 提交而取消正在跑的 run。

手动 `workflow_dispatch` 触发时使用 `github.run_id` 作为单独的 concurrency group，不跟 PR 的排队冲突。

## 7. 超时配置

Job 默认无限等，建议加 `timeout-minutes: 60` 防止远端卡死浪费资源。

## 8. 错误处理

| 场景 | 行为 |
|---|---|
| SSH 连接失败 | Job 直接 fail，不重试 |
| docker start 失败 | 远端命令返回非 0，Job fail |
| python ci.py 报错 | docker exec 返回非 0，Job fail |
| hello.txt 不存在 | docker cp + scp 失败，Job fail |

即 **ci.py 失败则门禁不通过**，PR 上会显示红 ✗。不做重试，不区分错误类型。

## 9. 产物保留

GitHub artifact 默认保留 **90 天**，到期自动清理。如需更长时间可在 workflow 中设置 `retention-days`。

## 10. GitHub Secrets 配置

需在 GitHub repo → Settings → Secrets and variables → Actions 中配置：

| Secret 名 | 值 |
|---|---|
| `DYNAMIC_CV_TEST_HOST` | 61.47.16.82 |
| `DYNAMIC_CV_TEST_USER` | z00896713 |
| `DYNAMIC_CV_TEST_PASSWORD` | a5-zhangkai |

## 11. Workflow 文件

路径：`.github/workflows/dynamic-cv-pipeline-tests.yml`

## 12. Agent

路径：`.claude/agents/ci-cd.md`（已创建）
