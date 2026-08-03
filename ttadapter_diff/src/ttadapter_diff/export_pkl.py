import traceback
from typing import Annotated
from pathlib import Path
from concurrent.futures import Future, as_completed

import typer
from pebble import ProcessPool, ProcessExpired

from .core.utils import get_cpu_count, serializer, target_in
from .core.offline_compile import offline_compile
from .schema import CompileResult, PythonException, SegfaultException, TritonKernelMetadata

def task(meta_file: Path, meta_dir: Path, result_dir: Path):
    with meta_file.open("rb") as f:
        meta: TritonKernelMetadata = serializer.load(f)
    try:
        result = offline_compile(meta)
    except Exception as e:
        print(meta.sys_path)
        raise e
    if result is not None:
        with target_in(meta_file, meta_dir, result_dir).open("wb") as f:
            serializer.dump(result, f)

def main(
    meta_dir: Annotated[Path, typer.Argument(help="编译元信息的储存目录", exists=True, file_okay=False, dir_okay=True, readable=True)], 
    result_dir: Annotated[Path, typer.Argument(help="编译结果输出目录", file_okay=False, dir_okay=True)],
    num_processes: Annotated[int, typer.Option("-n", "--num-processes", default_factory=lambda: max(get_cpu_count() - 1, 1), help="进程数", min=1)],
):
    tasks: list[Path] = []

    for meta_file in meta_dir.iterdir():
        if not meta_file.suffix == ".pkl":
            continue
        tasks.append(meta_file)

    if len(tasks) == 0:
        return

    result_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPool(max_workers=num_processes) as p:
        future_to_task: dict[Future, Path] = {
            p.schedule(task, args=[meta_file, meta_dir, result_dir]): meta_file
            for meta_file in tasks
        }
    
        for future in as_completed(future_to_task):
            meta_file = future_to_task[future]
            wrapped_error = None

            try:
                future.result()  # 获取结果，如果子进程崩溃会在这里抛出异常
                print(f"✅ [Success] 编译完成: {meta_file.name}")
            except ProcessExpired as error:
                print(f"💥 [FFI Segfault] 任务导致进程异常崩溃! 文件: {meta_file.name}")
                print(f"   退出码 (Exit code): {error.exitcode}, 异常进程 PID: {error.pid}")
                wrapped_error = SegfaultException(
                    message="Process expired abnormally (likely a segfault).",
                    exitcode=error.exitcode,
                    pid=error.pid
                )
            except Exception as error:
                print(f"⚠️ [Python Error] 任务执行失败: {meta_file.name}, 错误: {error}")
                tb_text = getattr(error, "traceback", None) or traceback.format_exc()
                wrapped_error = PythonException(
                    message=str(error),
                    traceback=tb_text
                )
            
            if wrapped_error is not None:
                with meta_file.open("rb") as f:
                    meta = serializer.load(f)
                with target_in(meta_file, meta_dir, result_dir).open("wb") as f:
                    serializer.dump(CompileResult(
                        ttadapter_str="",
                        meta=meta,
                        ssbuf_rc=0,
                        error=wrapped_error
                    ), f)