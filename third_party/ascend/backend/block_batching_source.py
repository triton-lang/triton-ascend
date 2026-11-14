# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Reparse a frozen frontend input at the ASTSource entry to the TTIR stage."""
import hashlib
from pathlib import Path

from triton._C.libtriton import ir
from triton.backends.compiler import Language
from triton.compiler.compiler import ASTSource


class FrozenSource(ASTSource):

    def __init__(self, source, path, fingerprint, digest):
        # Preserve launcher callbacks and the original signature in the caller only.
        self.__dict__.update(vars(source))
        self.path = str(Path(path).resolve())
        self.block_batching_source_hash = fingerprint
        self.digest = digest

    def hash(self):
        return self.block_batching_source_hash

    def to_wire(self):
        return dict(path=self.path, digest=self.digest, fingerprint=self.hash(), name=self.name, ext=self.ext,
                    language=self.language.name)

    @classmethod
    def from_wire(cls, record, directory):
        # Parse exactly the bytes we checked; a replaced/deleted source file must not
        # change the input between fingerprint verification and MLIR parsing.
        data = Path(record['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['digest']:
            raise ValueError('frozen frontend input changed')
        snapshot = Path(directory) / 'input.ttir'
        snapshot.write_bytes(data)
        source = cls.__new__(cls)
        source.name, source.ext, source.language = record['name'], record['ext'], Language[record['language']]
        source.path = str(snapshot)
        source.block_batching_source_hash, source.digest = record['fingerprint'], record['digest']
        return source

    def make_ir(self, target, options, codegen_fns, module_map, context):
        module = ir.parse_mlir_module(self.path, context)
        module.context = context
        return module
