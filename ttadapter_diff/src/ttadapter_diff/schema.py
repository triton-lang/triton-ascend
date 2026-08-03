from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import joblib

@dataclass(frozen=True, kw_only=True)
class TritonKernelMetadata:
    kernel_name: str
    module_name: str
    kernel_path: Path
    rel_kernel_path: Path
    signature: dict[str, str]
    constants: dict[str, Any]
    target: Any
    options: Any
    call_stack: list[str]
    sys_path: list[str]
    env: dict[str, str]
    sys_argv: list[str] = field(default_factory=list)

    @property
    def content(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "rel_kernel_path": self.rel_kernel_path.as_posix() if isinstance(self.rel_kernel_path, Path) else self.rel_kernel_path,
            "signature": self.signature,
            "constants": self.constants,
            "target": self.target,
            "options": self.options,
        }

    @property
    def content_hash(self) -> str | None:
        return joblib.hash(self.content, hash_name='md5')

@dataclass(frozen=True, kw_only=True)
class PythonException:
    message: str
    traceback: str

@dataclass(frozen=True, kw_only=True)
class SegfaultException:
    message: str
    exitcode: int | None = None
    pid: int | None = None

@dataclass(frozen=True, kw_only=True)
class CompileResult:
    ttadapter_str: str
    meta: TritonKernelMetadata
    ssbuf_rc: int
    error: PythonException | SegfaultException | None = None