"""Use-site dependency interpretation with branch joins and bounded loop fixed points."""
import ast
import inspect

from .. import config
from ..operations import STATIC_ARGUMENTS, is_language_builtin, operation, resolve
from ..syntax import parse
from .facts import FactResult, Fact, constant_value, merge


class Incomplete(Exception):
    pass


class Uses:

    def __init__(self, root, inputs):
        self.root = root
        self.inputs = inputs
        self.reasons = {}
        self.diagnostics = []
        self.remaining = config.ANALYSIS_NODE_BUDGET
        self.active = set()
        self.contexts = []

    def reject(self, value, reason, node):
        line = getattr(node, "lineno", 0)
        for dep in value.deps:
            self.reasons.setdefault(dep, set()).add((reason, line))

    def unknown(self, node, reason):
        self.diagnostics.append((reason, getattr(node, "lineno", 0)))
        raise Incomplete(reason)

    def opaque(self, node, reason, *values):
        """Reject only bounded dependencies; keep walking with an opaque value."""
        value = merge(*values)
        self.reject(value, reason, node)
        return Fact(value.deps, False, "unknown", True)

    def run(self):
        try:
            context, _ = self.function(self.root, self.inputs)
            return FactResult((context, ), True, tuple(self.diagnostics))
        except Incomplete:
            return FactResult((), False, tuple(self.diagnostics))

    def function(self, fn, inputs):
        if id(fn) in self.active:
            self.unknown(parse(fn), "RECURSIVE_JIT_CALL")
        self.active.add(id(fn))
        context = {"fn": fn, "inputs": dict(inputs), "locals": {}, "helpers": {}}
        scope = fn.get_capture_scope()
        env = dict(inputs)
        returns = []
        try:
            self.statements(parse(fn).body[0].body, env, scope, context, returns)
        finally:
            self.active.remove(id(fn))
        # The JIT frontend materializes scalar helper returns as tensors, even
        # when the returned expression was computed entirely from constexprs.
        return context, merge(*returns, runtime=True) if returns else Fact()

    def expression(self, node, env, scope, context):
        self.remaining -= 1
        if self.remaining < 0:
            self.unknown(node, "ANALYSIS_BUDGET_EXHAUSTED")
        if isinstance(node, ast.Constant):
            return constant_value(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                if env[node.id].kind == "callable":
                    self.unknown(node, "CALLABLE_VALUE_UNSUPPORTED")
                return env[node.id]
            obj = resolve(node, scope)

            import triton.language as tl

            if isinstance(obj, tl.constexpr):
                obj = obj.value
            if isinstance(obj, (int, float)):
                return constant_value(obj)
            if obj is not None or node.id in scope:
                if callable(obj) and not isinstance(obj, tl.dtype):
                    self.unknown(node, "CALLABLE_VALUE_UNSUPPORTED")
                return Fact(kind="static")
            self.unknown(node, "UNRESOLVED_NAME")
        if isinstance(node, (ast.Tuple, ast.List)):
            return merge(*(self.expression(n, env, scope, context) for n in node.elts), kind="tuple")
        if isinstance(node, ast.Attribute):
            import triton.language as tl

            obj = resolve(node, scope)
            if obj is not None:
                if callable(obj) and not isinstance(obj, tl.dtype):
                    self.unknown(node, "CALLABLE_VALUE_UNSUPPORTED")
                return Fact(kind="static")
            value = self.expression(node.value, env, scope, context)
            if node.attr not in ("dtype", "shape", "value"):
                if value.kind == "dtype":
                    return self.opaque(node, "ATTRIBUTE_SUMMARY_MISSING", value)
                self.unknown(node, "ATTRIBUTE_SUMMARY_MISSING")
            self.reject(value, "STATIC_ATTRIBUTE", node)
            kind = "dtype" if node.attr == "dtype" and value.kind.startswith(("ptr:", "block_ptr:")) else "static"
            return Fact(value.deps, False, kind, value.opaque)
        if isinstance(node, ast.Subscript):
            value = self.expression(node.value, env, scope, context)
            index = self.expression(node.slice, env, scope, context)
            self.reject(index, "STATIC_INDEX", node)
            return merge(value, index, runtime=value.runtime)
        if isinstance(node, ast.Slice):
            return merge(*(self.expression(n, env, scope, context)
                           for n in (node.lower, node.upper, node.step)
                           if n is not None))
        if isinstance(node, ast.BinOp):
            # Track every use; native Triton owns operand types and arithmetic.
            return merge(self.expression(node.left, env, scope, context),
                         self.expression(node.right, env, scope, context))
        if isinstance(node, ast.UnaryOp):
            return self.expression(node.operand, env, scope, context)
        if isinstance(node, ast.Compare):
            values = [self.expression(n, env, scope, context) for n in [node.left, *node.comparators]]
            return merge(*values, kind="bool")
        if isinstance(node, ast.BoolOp):
            values = [self.expression(n, env, scope, context) for n in node.values]
            for v in values:
                if v.opaque or not v.runtime:
                    self.reject(v, "CONTROL_SPECIALIZATION_PRESERVED", node)
            return merge(*values, kind="bool")
        if isinstance(node, ast.IfExp):
            condition = self.expression(node.test, env, scope, context)
            if condition.opaque or not condition.runtime:
                self.reject(condition, "CONTROL_SPECIALIZATION_PRESERVED", node)
            return merge(condition, self.expression(node.body, env, scope, context),
                         self.expression(node.orelse, env, scope, context))
        if isinstance(node, ast.Call):
            return self.call(node, env, scope, context)
        self.unknown(node, "EXPRESSION_SUMMARY_MISSING")

    def bind_required_arguments(self, node, op, parameters, args, kwargs):
        """Bind operations whose public parameters are all required."""
        supplied = dict(zip(parameters, args))
        if (len(args) > len(parameters) or kwargs.keys() & supplied.keys()
                or supplied.keys() | kwargs.keys() != set(parameters)):
            self.unknown(node, op.upper() + "_ARGUMENTS_UNSUPPORTED")
        supplied.update(kwargs)
        return supplied

    def call(self, node, env, scope, context):
        from triton.runtime.jit import JITFunction

        if any(isinstance(n, ast.Starred) for n in node.args) or any(k.arg is None for k in node.keywords):
            self.unknown(node, "STAR_CALL_UNSUPPORTED")
        args = [self.expression(n, env, scope, context) for n in node.args]
        kwargs = {k.arg: self.expression(k.value, env, scope, context) for k in node.keywords}
        target = node.func
        while isinstance(target, ast.Attribute):
            target = target.value
        # A local alias can shadow a captured builtin. Do not resolve that call
        # through globals; tensor methods are handled from the receiver below.
        obj = None if isinstance(target, ast.Name) and target.id in env else resolve(node.func, scope)
        op = operation(obj)
        if isinstance(obj, JITFunction) and op is None:
            try:
                bound = obj.signature.bind(*args, **kwargs)
            except TypeError:
                self.unknown(node, "JIT_CALLEE_BINDING_FAILED")
            for name, p in obj.signature.parameters.items():
                if name not in bound.arguments:
                    if p.default is inspect.Parameter.empty:
                        self.unknown(node, "JIT_CALLEE_DEFAULT_MISSING")
                    bound.arguments[name] = constant_value(p.default)
            child, result = self.function(obj, bound.arguments)
            # Revisited loop bodies must retain dependencies from earlier iterations.
            site = (node.lineno, node.col_offset)
            previous = context["helpers"].get(site)
            if previous and previous["inputs"] != child["inputs"]:
                inputs = {k: merge(previous["inputs"][k], child["inputs"][k]) for k in child["inputs"]}
                child, result = self.function(obj, inputs)
            context["helpers"][site] = child
            return result
        if op is None and obj is None and isinstance(node.func, ast.Attribute) and node.func.attr == "advance":
            base = self.expression(node.func.value, env, scope, context)
            if not base.kind.startswith("block_ptr:"):
                self.unknown(node, "ADVANCE_BASE_NOT_BLOCK_POINTER")
            args.insert(0, base)
            op = "advance"
        if op is None and isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            value = self.expression(node.func.value, env, scope, context)
            dtype_node = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "dtype"), None)
            dtype = resolve(dtype_node, scope) if dtype_node is not None else None
            kind = str(dtype)
            for v in [*args, *kwargs.values()]:
                self.reject(v, "STATIC_DTYPE", node)
            return Fact(value.deps, value.runtime, kind, value.opaque
                        or any(v.opaque for v in [*args, *kwargs.values()]))
        if op is None:
            if is_language_builtin(obj):
                return self.opaque(node, "OPERATION_SUMMARY_MISSING", *args, *kwargs.values())
            if ((obj is float or obj is int or obj is bool) and not kwargs and len(args) == 1 and not args[0].opaque
                    and args[0].kind in ("int", "float", "bool", "str")):
                return self.opaque(node, "OPERATION_SUMMARY_MISSING", *args)
            self.unknown(node, "OPERATION_SUMMARY_MISSING")
        if op == "builtin_min" and (len(args) < 2 or kwargs.keys() - {"propagate_nan"}):
            # Iterable/key/default forms have no runtime tensor summary.
            self.unknown(node, "OPERATION_SUMMARY_MISSING")
        all_values = [*args, *kwargs.values()]
        value = merge(*all_values)
        if op in ("static_range", "static_assert", "pointer_type"):
            self.reject(value, "STATIC_" + op.upper(), node)
        if op == "range":
            for attribute in args[3:]:
                self.reject(attribute, "STATIC_LOOP_ATTRIBUTE", node)
            for name, attribute in kwargs.items():
                if name not in ("arg1", "arg2", "step", "start", "end"):
                    self.reject(attribute, "STATIC_LOOP_ATTRIBUTE", node)
        positions, names = STATIC_ARGUMENTS.get(op, ((), ()))
        for i in positions:
            if i < len(args):
                self.reject(args[i], "STATIC_" + op.upper(), node)
        for name in names:
            if name in kwargs:
                self.reject(kwargs[name], "STATIC_" + op.upper(), node)
        if op == "make_block_ptr":
            parameters = ("base", "shape", "strides", "offsets", "block_shape", "order")
            supplied = self.bind_required_arguments(node, op, parameters, args, kwargs)
            base = supplied["base"]
            # Parent shape/strides and offsets are runtime metadata; only the
            # block shape and order determine the static block pointer type.
            element = base.kind.removeprefix("ptr:") if base.kind.startswith("ptr:") else "unknown"
            if element in ("bool", "int1"):
                element = "int8"  # Native make_block_ptr promotes int1 pointers.
            return Fact(value.deps, True, "block_ptr:" + element, value.opaque)
        if op == "advance":
            supplied = self.bind_required_arguments(node, op, ("base", "offsets"), args, kwargs)
            base = supplied["base"]
            if not base.kind.startswith("block_ptr:"):
                self.unknown(node, "ADVANCE_BASE_NOT_BLOCK_POINTER")
            return Fact(value.deps, True, base.kind, value.opaque)
        if op == "dot":
            parameters = ("input", "other", "acc", "input_precision", "allow_tf32", "max_num_imprecise_acc",
                          "out_dtype")
            supplied = set(parameters[:len(args)]) | kwargs.keys()
            if (len(args) > len(parameters) or kwargs.keys() - set(parameters)
                    or kwargs.keys() & set(parameters[:len(args)]) or not {"input", "other"} <= supplied):
                self.unknown(node, "DOT_ARGUMENTS_UNSUPPORTED")
            # Dot is a runtime tensor contraction, not a pointwise operation.
            # Its shape producers and instruction attributes stay static. Keep
            # data dependencies, but do not inherit the lhs element type: fp16
            # can accumulate into fp32, and int8 into int32. Fact.kind is not a
            # tensor type system; native Triton owns dot typing and validation.
            return Fact(value.deps, True, "unknown", value.opaque)
        if op in ("program_id", "num_programs", "arange"):
            return Fact(value.deps, True, "int", value.opaque)
        if op == "load":
            pointer = args[0] if args else kwargs.get("pointer", Fact(kind="unknown"))
            other = args[2] if len(args) > 2 else kwargs.get("other", Fact())
            # Address dependence does not change the loaded element's type.
            return Fact(other.deps, True,
                        pointer.kind.removeprefix("ptr:").removeprefix("block_ptr:"), pointer.opaque or other.opaque)
        if op == "cast":
            dtype_node = node.args[1] if len(node.args) > 1 else next(
                (k.value for k in node.keywords if k.arg == "dtype"), None)
            dtype = resolve(dtype_node, scope) if dtype_node is not None else None
            if isinstance(dtype_node, ast.Call) and operation(resolve(dtype_node.func, scope)) == "pointer_type":
                element = resolve(dtype_node.args[0], scope)
                return Fact(runtime=True, kind="ptr:" + str(element), opaque=value.opaque)
            kind = str(dtype)
            return Fact(value.deps, value.runtime, kind, value.opaque)
        if op == "where":
            condition = args[0] if args else kwargs.get("condition", Fact())
            x = args[1] if len(args) > 1 else kwargs.get("x", Fact(kind="unknown"))
            y = args[2] if len(args) > 2 else kwargs.get("y", Fact(kind="unknown"))
            return merge(condition, x, y, runtime=True, kind=y.kind)
        # Unlike tl.minimum, builtin min preserves constexpr-only evaluation.
        return Fact(value.deps, value.runtime or op not in ("static_range", "range", "pointer_type", "builtin_min"),
                    value.kind, value.opaque)

    def assign(self, target, value, env, node):
        if not isinstance(target, ast.Name):
            self.unknown(node, "ASSIGNMENT_SUMMARY_MISSING")
        env[target.id] = value

    def statements(self, statements, env, scope, context, returns):
        import triton.language as tl

        for node in statements:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = self.expression(node.value, env, scope, context)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    self.assign(target, value, env, node)
                if isinstance(node, ast.AnnAssign):
                    if resolve(node.annotation, scope) is not tl.constexpr:
                        self.unknown(node, "LOCAL_ANNOTATION_UNSUPPORTED")
                    if value.opaque:
                        self.reject(value, "OPAQUE_VALUE_IN_LOCAL_CONSTEXPR", node)
                    elif value.runtime:
                        self.reject(value, "RUNTIME_VALUE_IN_LOCAL_CONSTEXPR", node)
                    context["locals"][node.lineno] = context["locals"].get(node.lineno, frozenset()) | value.deps
            elif isinstance(node, ast.AugAssign):
                value = self.expression(ast.copy_location(ast.BinOp(node.target, node.op, node.value), node), env,
                                        scope, context)
                self.assign(node.target, value, env, node)
            elif isinstance(node, ast.Expr):
                self.expression(node.value, env, scope, context)
            elif isinstance(node, ast.Return):
                if node.value:
                    returns.append(self.expression(node.value, env, scope, context))
            elif isinstance(node, ast.If):
                condition = self.expression(node.test, env, scope, context)
                if condition.opaque or not condition.runtime:
                    self.reject(condition, "CONTROL_SPECIALIZATION_PRESERVED", node)
                left, right = dict(env), dict(env)
                self.statements(node.body, left, scope, context, returns)
                self.statements(node.orelse, right, scope, context, returns)
                for name in left.keys() | right.keys():
                    env[name] = merge(left.get(name, Fact()), right.get(name, Fact()))
            elif isinstance(node, (ast.For, ast.While)):
                before = dict(env)
                # Bound fixed-point analysis of loop-carried dependencies.
                for _ in range(16):
                    previous = dict(env)
                    if isinstance(node, ast.For):
                        value = self.expression(node.iter, env, scope, context)
                        op = operation(resolve(node.iter.func, scope)) if isinstance(node.iter, ast.Call) else None
                        if op not in ("range", "static_range"):
                            self.unknown(node, "LOOP_ITERATOR_UNSUPPORTED")
                        self.assign(node.target, Fact(value.deps, op != "static_range", "int", value.opaque), env, node)
                    else:
                        condition = self.expression(node.test, env, scope, context)
                        if condition.opaque or not condition.runtime:
                            self.reject(condition, "CONTROL_SPECIALIZATION_PRESERVED", node)
                    self.statements(node.body, env, scope, context, returns)
                    # Backedges merge dependencies, not a last-assignment map.
                    for name in before.keys() | previous.keys() | env.keys():
                        values = [m[name] for m in (before, previous, env) if name in m]
                        env[name] = merge(*values)
                    if env == previous:
                        break
                else:
                    self.unknown(node, "LOOP_ANALYSIS_NOT_CONVERGED")
                self.statements(node.orelse, env, scope, context, returns)
            elif isinstance(node, (ast.Pass, ast.Break, ast.Continue)):
                pass
            else:
                self.unknown(node, "STATEMENT_SUMMARY_MISSING")
