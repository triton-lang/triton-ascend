"""Trace the pinned libdevice SIMD body instead of assigning pow a guessed cost."""
import json
from pathlib import Path

from backend_cost_model import Analyzer, Unsupported, known
from cpu_resource_gate import sha256
from vv_contention_model import DependencySchedule

DATA = Path(__file__).with_name("vv_library_data")
POW = "_mlir_ciface_pow_fp32.extracted"


class LibraryModel:

    def __init__(self, table, assumptions, vector_factory, evidence):
        self.provenance = json.loads((DATA / "provenance.json").read_text())
        if not evidence or evidence["sha256"] != self.provenance["libdevice_sha256"]:
            raise Unsupported("pow needs matching pinned libdevice evidence")
        path = DATA / "pow_fp32.ll"
        if sha256(path) != self.provenance["llvm_sha256"]:
            raise Unsupported("extracted pow LLVM changed")
        self.analyzer = Analyzer(path.read_text(), table, assumptions, vector_factory=vector_factory)
        self.wrapper_steps = self.provenance["descriptor_scalar_steps"]

    def pow(self, args):
        if len(args) != 15:
            raise Unsupported("unsupported pow flattened memref ABI")
        sizes = [int(known(args[i], "unknown pow extent")) for i in (3, 8, 13)]
        strides = [known(args[i], "unknown pow stride") for i in (4, 9, 14)]
        if len(set(sizes)) != 1 or not 0 <= sizes[0] < 2**31 or strides != [1, 1, 1]:
            raise Unsupported("pow library supports equal contiguous 1D fp32 extents")
        n = sizes[0]
        tail = ("stack", "pow_tail", n % 64)
        result = self.analyzer.trace(POW, [n // 64, tail, None, None, None], 0, initial_memory={tail: n % 64})
        return result


class LibrarySchedule(DependencySchedule):

    def __init__(self, assumptions, library):
        super().__init__(assumptions)
        self.library = library

    def extra_call(self, name, args):
        if name != "pow_fp32":
            return False
        if self.library is None:
            raise Unsupported("missing pow library model")
        trace = self.library.pow(args)
        # Descriptor setup is straight-line wrapper work. Retain an explicit
        # scalar upper estimate; final ISA optimization can remove some of it.
        for _ in range(self.library.wrapper_steps):
            self.scalar_step()
        self.vf(trace)
        self.counts["library_calls:pow_fp32"] += 1
        self.counts["cost_source:library_descriptor_scalar_upper_estimate"] += self.library.wrapper_steps
        return True
