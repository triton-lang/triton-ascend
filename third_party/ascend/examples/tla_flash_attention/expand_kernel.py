"""Expand named source stages; emit ordinary TLA Python, with no runtime macros.

Only six static pipeline phases and calls to ``inline_*`` are expanded. The
expander does not reassociate expressions, move operations or tune constants.
Run this file without an NPU to materialize every kernel and check its AST
against the frozen measured implementation's digest.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "long_d128": ("phase4_kqt_dualkv_native", False, False),
    "long_d64": ("phase4_pv2_dualkv", True, False),
    "short_causal_d128": ("phase4_tasks_mixed", False, True),
    "short_dense_d128": ("phase4_kqt_native", False, False),
    "short_causal_d64": ("phase4_pv2_compact", True, True),
    "short_dense_d64": ("phase4_pv2", True, False),
}


class Substitute(ast.NodeTransformer):

    def __init__(self, arguments):
        self.arguments = arguments

    def visit_Name(self, node):
        return copy.deepcopy(self.arguments.get(node.id, node))

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            return node.body if node.test.value else node.orelse
        return node


class RotateSlots(ast.NodeTransformer):
    """Rotate physical rings and their ready/free flags together."""

    def __init__(self, phase, double_pv):
        self.phase = phase
        self.double_pv = double_pv

    def visit_Name(self, node):
        match = re.fullmatch(r"(s|p|pf|alpha)([0-2])", node.id)
        if match:
            period = 2 if match[1] == "s" else 3
            node.id = match[1] + str((int(match[2]) + self.phase) % period)
        match = re.fullmatch(r"(pv_ub|k_l1_|v_l1_)([01])", node.id)
        if match:
            node.id = match[1] + str((int(match[2]) + self.phase) % 2)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.func) in ("tla.sync.block_wait_flag", "tla.sync.block_set_flag"):
            rings = [(0, 4), (10, 11), (1, 5, 13)]
            if self.double_pv:
                rings += [(2, 6), (9, 12)]
            flag = node.args[-1].value
            for ring in rings:
                if flag in ring:
                    node.args[-1].value = ring[(ring.index(flag) + self.phase) % len(ring)]
                    break
        return node


class Expand(ast.NodeTransformer):

    def __init__(self, macros, double_pv):
        self.macros = macros
        self.double_pv = double_pv

    def visit_Expr(self, node):
        call = node.value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            return self.generic_visit(node)
        if call.func.id not in self.macros:
            return self.generic_visit(node)
        macro = self.macros[call.func.id]
        assert not call.keywords and len(call.args) == len(macro.args.args)
        arguments = dict(zip((a.arg for a in macro.args.args), call.args))
        body = ast.Module(body=copy.deepcopy(macro.body), type_ignores=[])
        body = Substitute(arguments).visit(body)
        return self.visit(body).body

    def visit_For(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.iter) != "pipeline_phases(6)":
            return node
        result = []
        for phase in range(6):
            body = ast.Module(body=copy.deepcopy(node.body), type_ignores=[])
            body = RotateSlots(phase, self.double_pv).visit(body)
            assert ast.unparse(body.body[0]) == "t = step + 0"
            body.body[0].value.right.value = phase
            result += body.body
        return result


def expanded_function(variant):
    name, double_pv, _ = VARIANTS[variant]
    tree = ast.parse((HERE / "kernel.py").read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    macros = {k: v for k, v in functions.items() if k.startswith("inline_")}
    fn = Expand(macros, double_pv).visit(functions["fa_forward_" + name])
    fn.decorator_list = [ast.Name(id="TLAJITFunction", ctx=ast.Load())]
    return ast.fix_missing_locations(fn)


def ast_digest(fn):
    return hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()


def render(variant, frontend="compat"):
    """Emit the measured AST or the direct upstream interface equivalent."""
    assert frontend in ("compat", "native")
    _, _, tail = VARIANTS[variant]
    fn = expanded_function(variant)
    expected = json.loads((HERE / "evidence/ast_equivalence.json").read_text())[variant]
    if ast_digest(fn) != expected["ast_sha256"]:
        raise RuntimeError(f"{variant}: expanded kernel changed; revalidate before timing")
    if frontend == "compat":
        extension = "tla_native_tail_ext" if tail else "tla_native_ext"
        imports = f"import {extension} as ext\n"
    else:
        # A local alias keeps the arithmetic and schedule unchanged. Both
        # interfaces emit the same existing AVE/HIVM operations.
        imports = "import native_ops as ext\n"
    imports += "import triton.language as tl\nimport triton.language.extra.tla as tla\n"
    imports += "from triton.language.extra.tla.jit import TLAJITFunction\n"
    return imports + "\n" + ast.unparse(fn) + "\nfa_forward = " + fn.name + "\n"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "generated")
    parser.add_argument("--frontend", choices=("compat", "native"), default="compat")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        source = render(variant, args.frontend)
        compile(source, variant, "exec")
        (args.output / (variant + ".py")).write_text(source)
        print(variant, "AST matches measured kernel")
