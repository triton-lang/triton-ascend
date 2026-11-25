"""Content identity for an Ascend compilation toolchain."""
import hashlib
import json
import os
from pathlib import Path
import shutil


class _FileChanged(RuntimeError):
    pass


def _stat_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _digest_at_identity(path, identity):
    # Stat equality does not prove content equality: same-size in-place writes
    # can share timestamps. Always read bytes for each fingerprint lookup.
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        if _stat_identity(os.fstat(stream.fileno())) != identity:
            raise _FileChanged(path)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if _stat_identity(os.fstat(stream.fileno())) != identity:
            raise _FileChanged(path)
    if _stat_identity(os.stat(path)) != identity:
        raise _FileChanged(path)
    return digest.hexdigest()


def _file_identity(path):
    for _ in range(3):
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file():
            raise OSError(f"toolchain component is not a file: {resolved}")
        identity = _stat_identity(resolved.stat())
        try:
            digest = _digest_at_identity(str(resolved), identity)
            if _stat_identity(resolved.stat()) == identity and Path(path).resolve(strict=True) == resolved:
                return str(resolved), digest
        except _FileChanged:
            continue
    raise RuntimeError(f"toolchain component changed while computing its identity: {path}")


def toolchain_fingerprint(compiler, reader, ir_tools, libdevice, environment):
    """Hash selected tool contents and their packaged template/library payloads.

    Pass reader=None for a frontend that consumes IR directly.
    File metadata guards against changes during a read; it never caches content
    digests and is not part of the resulting key.
    This runs for compilation cache lookup, not hot replacement of loaded kernels.
    """
    frontend = Path(compiler).resolve(strict=True)
    components = [("frontend", frontend), ("libdevice", libdevice)]
    if reader is not None:
        components.append(("reader", reader))
    components.extend((f"ir-tool-{index}", path) for index, path in enumerate(ir_tools))
    wrapper = environment.get("BISHENG_CAPTURE_WRAPPER")
    real = environment.get("TRITON_REAL_BISHENGIR_COMPILE")
    if wrapper and real and Path(wrapper).resolve() == frontend:
        frontend = Path(real).resolve(strict=True)
        components.append(("capture-real-frontend", frontend))

    library_root = frontend.parent.parent / "lib"
    if library_root.is_dir():
        for path in sorted(library_root.rglob("*")):
            if path.is_file() and (path.suffix in (".bc", ".ll", ".so") or ".so." in path.name):
                components.append(("frontend-lib/" + path.relative_to(library_root).as_posix(), path))

    downstream_directory = environment.get("BISHENG_INSTALL_PATH")
    for name in ("bisheng", "ccec", "hivmc", "hivmc-a5", "ld.lld"):
        selected = shutil.which(name, path=environment.get("PATH", ""))
        if selected:
            components.append(("downstream-path/" + name, selected))
        if downstream_directory:
            explicit = Path(downstream_directory) / name
            if explicit.is_file():
                components.append(("downstream-explicit/" + name, explicit))

    entries = [(role, *_file_identity(path)) for role, path in sorted(components, key=lambda item: item[0])]
    payload = json.dumps(["ascend-toolchain-v1", entries], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
