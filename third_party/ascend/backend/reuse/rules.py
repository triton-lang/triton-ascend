"""Structural proof for masked grid-stride and program-local elementwise loops."""
import ast

from .model import LaunchRecipe
from .operations import operation, resolve
from .syntax import parse


def _grid_affine(expression):
    """Accept integer grid-only expressions affine in program IDs.

    Return whether an expression contains a program ID. Actual coverage and
    every original intermediate are checked against the invocation's grid.
    """
    op, *operands = expression
    if op == "lit" and type(operands[0]) is int:
        return False
    if op == "num_programs":
        return False
    if op == "program_id":
        return True
    if op in ("add", "sub", "mul"):
        left, right = (_grid_affine(value) for value in operands)
        if op != "mul" or not (left and right):
            return left or right
    raise ValueError("unsupported grid-stride expression")


def match_grid_stride(fn):
    scope = fn.get_capture_scope()
    env = {name: ("arg", i) for i, name in enumerate(fn.arg_names)}
    pointers, stores, accesses = set(), set(), []
    stored = False
    loop_writes, loop_defined = set(), set()

    def expr(node):
        if isinstance(node, ast.Name):
            if node.id in loop_writes and node.id not in loop_defined:
                raise ValueError("loop-carried value")
            return env[node.id]
        if isinstance(node, ast.Constant):
            return ("lit", node.value)
        if isinstance(node, ast.BinOp):
            op = {ast.Add: "add", ast.Mult: "mul", ast.Sub: "sub"}[type(node.op)]
            return (op, expr(node.left), expr(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Lt):
            return ("lt", expr(node.left), expr(node.comparators[0]))
        if isinstance(node, ast.Call):
            op = operation(resolve(node.func, scope))
            args = tuple(expr(n) for n in node.args)
            kws = {k.arg: expr(k.value) for k in node.keywords}
            if op in ("program_id", "num_programs"):
                axes = args or (kws.get("axis"), )
                if len(axes) != 1 or axes[0] not in (("lit", 0), ("lit", 1), ("lit", 2)):
                    raise ValueError("unsupported program axis")
                if set(kws) - {"axis"} or (args and kws):
                    raise ValueError("unsupported program arguments")
                return (op, axes[0][1])
            if op == "arange" and len(args) == 2 and not kws:
                return (op, *args)
            if op == "cdiv" and len(args) == 2 and not kws:
                return (op, *args)
            if op == "load" and not stored:
                if len(args) != 1 or set(kws) - {"mask", "other"}:
                    raise ValueError("load effects")
                accesses.append((args[0], kws.get("mask"), False))
                return ("data", len(accesses))
        raise ValueError("unmodeled expression")

    def assign(node):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ValueError("non-simple assignment")
        env[node.targets[0].id] = expr(node.value)
        loop_defined.add(node.targets[0].id)

    def has_schedule(value):
        if value in (offset, ("block", ), ("arg", tile)):
            return True
        if not local_loop and value[0] in ("program_id", "num_programs"):
            return True
        if bound_slot is not None and value == bound:
            return True
        return any(has_schedule(v) for v in value[1:] if isinstance(v, tuple))

    def add_terms(value):
        if value[0] == "add":
            return add_terms(value[1]) + add_terms(value[2])
        return [value]

    def split_address(address):
        terms = add_terms(address)
        for term in add_terms(offset):
            terms.remove(term)
        if not terms or terms[0][0] != "arg":
            raise ValueError("missing base pointer")
        pointer = terms.pop(0)[1]
        base = ("lit", 0)
        for term in terms:
            base = term if base == ("lit", 0) else ("add", base, term)
        if has_schedule(base) or (not local_loop and base != ("lit", 0)):
            raise ValueError("base depends on tile scheduling")
        return pointer, base

    try:
        body = parse(fn).body[0].body
        loops = [n for n in body if isinstance(n, ast.For)]
        if len(loops) != 1 or loops[0] is not body[-1]:
            return None
        loop = loops[0]
        for n in body[:-1]:
            assign(n)
        if not isinstance(loop.iter, ast.Call) or operation(resolve(loop.iter.func, scope)) != "range":
            return None
        if loop.iter.keywords or not 1 <= len(loop.iter.args) <= 3 or loop.orelse or not isinstance(
                loop.target, ast.Name):
            return None
        loop_args = tuple(expr(n) for n in loop.iter.args)
        if len(loop_args) == 1:
            start, bound, step = ("lit", 0), loop_args[0], ("lit", 1)
        elif len(loop_args) == 2:
            start, bound, step = *loop_args, ("lit", 1)
        else:
            start, bound, step = loop_args
        local_loop = start == ("lit", 0)
        if not local_loop and (not _grid_affine(start) or _grid_affine(step)):
            return None
        env[loop.target.id] = ("block", )
        # Every value assigned in the loop must be defined before it is read
        # in that iteration. Otherwise changing the tile changes its history.
        loop_writes = {
            node.id
            for statement in loop.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        loop_defined = {loop.target.id}
        store_values = []
        for n in loop.body:
            if isinstance(n, ast.Assign):
                assign(n)
            elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
                call = n.value
                if operation(resolve(call.func, scope)) != "store" or len(call.args) != 2:
                    return None
                kws = {k.arg: expr(k.value) for k in call.keywords}
                if set(kws) != {"mask"}:
                    return None
                accesses.append((expr(call.args[0]), kws["mask"], True))
                store_values.append(expr(call.args[1]))
                stored = True
            else:
                return None
        if not accesses or not store_values:
            return None
        _, mask, _ = accesses[0]
        if mask is None or mask[0] != "lt" or mask[2][0] != "arg":
            return None
        offset = mask[1]
        if offset[0] != "add" or offset[2][:2] != ("arange", ("lit", 0)):
            return None
        tile_arg = offset[2][2]
        if tile_arg[0] != "arg" or offset[2] != ("arange", ("lit", 0), tile_arg):
            return None
        tile = tile_arg[1]
        if mask[2] == tile_arg:
            return None
        bound_slot = None
        if local_loop and step == tile_arg:
            # range(0, N, B): changing B also changes the loop step in the binary.
            if bound != mask[2] or offset[1] != ("block", ):
                return None
        else:
            if local_loop and step != ("lit", 1):
                return None
            if offset[1] != ("mul", ("block", ), tile_arg):
                return None
            if bound[0] == "arg":
                bound_slot = bound[1]
                if mask[2] == bound or fn.params[bound_slot].is_constexpr:
                    return None
            elif not local_loop or bound != ("cdiv", mask[2], tile_arg):
                return None
        if not fn.params[tile].is_constexpr:
            return None
        program_offsets = {}
        for address, access_mask, write in accesses:
            if access_mask != mask:
                return None
            pointer, base = split_address(address)
            if pointer in program_offsets and program_offsets[pointer] != base:
                return None
            program_offsets[pointer] = base
            pointers.add(pointer)
            if write:
                stores.add(pointer)
        if any(has_schedule(v) for v in store_values):
            return None
        rule = "MaskedProgramLocalElementwiseV1" if local_loop else "MaskedGridStrideElementwiseV2"
        return LaunchRecipe(tile, mask[2][1], bound_slot, tuple(sorted(pointers)), tuple(sorted(stores)), rule=rule,
                            local_loop=local_loop, program_offsets=tuple(sorted(program_offsets.items())),
                            grid_start=start, grid_step=step)
    except (KeyError, ValueError, IndexError, TypeError):
        return None
