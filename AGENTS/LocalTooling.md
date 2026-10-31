# Local Tooling, Scripts & Build Environment (LocalTooling.md)

Reference for local-only tooling. The root `AGENTS.md` keeps only the high-frequency entries
(full build, UT suite, single-UT command) and points here for the rest — so this file is opened
on demand, never carried in every session's context.

---

## 1. Where a script lives

Exactly two homes:

| Location | Rule |
| --- | --- |
| `~/personal/` | Only what must work **without** a repo checkout (`build_ta.sh`, and its own copy of this file is *not* maintained). A machine that builds this project is known to have `~/personal/`, but `~/personal/` is not where repo docs live. |
| Repo (the `localhack` layer) | Everything else: `AGENTS/**` docs (incl. this file), `python/update_ut.py`, the in-tree patch, and the scripts in §2. Local-only: `localhack` = `ukxrlwzt/0` **is an ancestor of the working chain** (its change id is divergent — several copies exist, `@origin` is ahead — so always address it as `ukxrlwzt/0` / `localhack`, never bare `ukxrlwzt`). |

This file therefore lives at `AGENTS/LocalTooling.md` in the repo; edit it there, not in `~/personal/`.

A `~/personal/` script may move into the repo **only if it is path-independent** (no hardcoded
`REPO` / toolchain path).

Where the local-tooling commit sits: **inside the `localhack` layer** — a child of `ukxrlwzt/0`,
below the refactor commits, so nothing tool-related is reviewable or pushable upstream.

**Doc fold-back rule (ends the recurring `AGENTS.md` rebase conflicts)**: generic doc content
(root `AGENTS.md`, shared-helper entries of `AGENTS/Utils.md`, this file) lives in the `localhack`
layer; a feature commit carries only its own pass doc. When a feature change must touch a generic
doc, fold the delta back before rebasing anything:
`jj squash --from @ --into ukxrlwzt/0 <path>` — both work lines then inherit the merged content
through the common ancestor, and the other line's doc delta resolves as "take destination side".
Pass-specific entries never go into root `AGENTS.md`; they belong in the pass doc of the line that
introduced them. A path that does **not yet exist** at `ukxrlwzt/0` (a new local-only doc) instead
produces an add/add conflict chain down to every descendant that carries an older copy: write the
new content at the layer itself, then `jj restore --from ukxrlwzt/0 <path>` at the earliest
conflicted descendant; after the squash re-check `jj log -r 'conflicts()'` for the whole chain.

---

## 2. Script index

| Script | Location | Frequency | Path-independent | Purpose |
| --- | --- | --- | --- | --- |
| `build_ta.sh` | `~/personal/` | every session | no (hardcoded paths) | Configure + full build; supplies the `-D__LLVM_MAJOR_VERSION_22_COMPATIBLE__` patch. Mandatory after a fresh configure. |
| `run_dcv_ut.py` | repo root | every session | yes | `lit` stand-in: walks the `// RUN:` lines of the DCV UT tree. Repo root from `$TRITON_ASCEND_ROOT`, else cwd, else the default checkout; `FileCheck` from `$FILECHECK` or the local toolchain glob; `triton-opt` from `$TRITON_OPT` or `build/cmake.*/bin/`. |
| `restore_sources.py` | repo root | on demand | yes | Undo the in-tree patch (`build_ta.sh` leaves tracked files dirty). |
| `dump_dcv_ut.py` | repo root | on refactor / equivalence checks | yes | Dump the whole-file `triton-opt` output of every DCV UT into a directory (RUN line minus `| FileCheck`). Two builds + `diff -rq` proves a change is output-neutral, including drift no `CHECK` line asserts. |
| `split_cpp_sections.py` | repo root | on refactors only | yes (repo-relative paths, run from the repo root) | Split one monolithic pass `.cpp` into phase TUs by whole-section moves. Spec-driven; `--dry-run` diffs against the current files, `--source <pristine.cpp>` replays a finished split. Worked example: the `SplitIfByBlockId` P3 split. |

---

## 3. Environment facts

- **Toolchain layout**: the LLVM build used here is `~/.triton/llvm/llvm-ubuntu-x64` (stable symlink to the pinned `llvm-f6ded0be-*-ubuntu-x64`; provides `FileCheck`, `clang`, `llvm-lit`); `triton-opt` comes from the repo's own `build/cmake.linux-x86_64-cpython-3.11/bin/`.
- **No LLVM/MLIR source checkout exists here**: `~/.triton/llvm/llvm-*-ubuntu-x64` ships headers + libs only, so `lib/IR/Operation.cpp` is nowhere on disk -> confirm a library precondition (e.g. `Operation::isBeforeInBlock`'s "within the same parent block") from the installed header (`include/mlir/IR/Operation.h`) instead of hunting for the implementation.
- **Never export the LLVM bin dir globally**: it also ships `clang`/`clang++`/`FileCheck`/`llvm-lit`, so a `~/.bashrc` export shadows the system compiler for every other project. Prepend it per command.
- **Agent tool shells do not source `~/.bashrc`**: they are non-login, non-interactive, with `BASH_ENV` empty, so exports/aliases from `~/.bashrc` do not apply -> put environment setup in the command itself (or in the shell that launched the agent). Probe before relying on inherited setup: `echo $PATH`, `shopt -q login_shell`.
- **The commit-gate `clang-format` is v19.1.6, not the system one**: `/usr/bin/clang-format` is 23.1.2 and the LLVM toolchain dir ships no `clang-format` at all; the binary that faithfully reproduces `pre-commit run clang-format` (mirrors-clang-format v19.1.6) is the pre-commit cache copy — `~/.cache/pre-commit/repo*/py_env-python3/lib/python3.11/site-packages/clang_format/data/bin/clang-format` — use it for manual `--dry-run --Werror` checks; v23 disagrees with v19 (include ordering) and misattributes violations.
- **`--debug-only=<DEBUG_TYPE>` 在本构建可用（`LLVM_DEBUG` 未编译掉）**: 二进制虽被 `safe_compile.cmake` strip，但 `DebugFlag` 仍生效，DCV 各 TU 的 `LDBG`/`LOG_DEBUG` 能直接打（`--debug-only=memory-effects-tracker` / `--debug-only=pack-memref` 实测有输出）-> 验证“无 IR 可观测”的内部行为（依赖图、mem slot 状态、聚合效果集）时，首选“临时加一行 LDBG + 重编单个 TU + 跑 `--debug-only`”，验完删掉再重编（PASS 认证的是当时字节）。
- **仓库根的 `.env` 是字符设备（`crw-rw-rw-`，1,3 = `/dev/null`），不是普通文件**: jj 无法快照设备节点，因此一个新建的工作副本提交会在 `jj status` 里报 `D .env`、`git status` 报 ` M .env`（`git diff` 为空）-> 这是快照差异，不是 build/`restore_sources.py` 弄脏了树，**不要 `jj restore .env`**（那会把设备节点换成空文件，弄坏依赖它的本地调试配置），也不要为了“干净”去动它。
- **`.pi/` 在本仓库被 `.git/info/exclude` 忽略**: `jj diff --summary -r @` 与 `git status` 都不列 `.pi/curr_requirements.md`、`.pi/last_review.md`，所以本仓库**不需要**把 `.pi` 状态搬去 `localhack`（全局 AGENTS.md 那条“jj 会快照未跟踪文件、`.pi/last_review.md` 已在 `@` 里”的前提在此不成立）。
- **`--add_dynamic_cv_pipeline` 对本地裁剪 IR / 真实 kernel 输入常直接 fallback，别拿它当验证入口**: 产物里 `rc = 2`（与加不加 `compile-on-910-95=1` 无关）= `Kernel not applicable for dynamic CV pipeline`（无 matmul / 已 scope 化 / 不支持的形态），整条管线被跳过，看起来“改了没影响” -> 验证单个 DCV pass 要走显式 pass 链（`--ssbuf-standardize-op-pack-memref --plan-compute-block --compute-block-opt`）；fallback 原因需 `--debug-only=add-dynamic-cv-pipeline` 才看得到（该 pass 装了 `ScopedDiagnosticHandler`，`emitWarning` 在 release 下被吞）。
- **`.reviewignore` (repo root, `localhack` layer)**: the code-review extension reads it to exclude paths from review, the format precheck and the settlement-gate hash. It lists the **17 upstream-triton paths** touched by `build_ta.sh`'s `triton-ascend-3.6.0.patch`, deliberately explicit: a directory or basename rule (`python/`, `CMakeLists.txt`) would also hide this repo's own files under `third_party/ascend/`. Everything else — `third_party/ascend/lib|include|unittest` included — stays reviewed, and the report prints `已排除 N 个文件` so a stale rule is visible.
