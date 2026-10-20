"""Same-process JIT snapshots. Original source, params and caches are never edited."""
import ast
import copy
import inspect
import threading
import types
from collections import defaultdict

from triton.compiler.compiler import ASTSource
from triton.runtime.jit import JITFunction, KernelParam

from . import config
from .identity import digest, typed


class ReuseASTSource(ASTSource):

    def hash(self):
        return digest((super().hash(), self.fn.reuse_identity, typed(self.constants), typed(self.attrs),
                       config.SCHEMA_VERSION, config.RULE_VERSION, config.ABI_VERSION))


class FrozenJITFunction(JITFunction):

    def get_capture_scope(self):
        return self.__globals__

    def create_binder(self):
        result = super().create_binder()
        self.ASTSource = ReuseASTSource
        return result

    @classmethod
    def from_plan(cls, plan):
        original = plan.fn
        tree = copy.deepcopy(original.parse())
        scope = dict(original.get_capture_scope())
        local_names = set(original.arg_names) | {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        helper_names = {}
        for site, child in plan.helpers:
            name = f"_reuse_helper_{site[0]}_{site[1]}"
            while name in scope or name in local_names:
                name += "_"
            scope[name] = cls.from_plan(child)
            helper_names[site] = name

        class Rewrite(ast.NodeTransformer):

            def visit_Call(self, node):
                self.generic_visit(node)
                site = (node.lineno, node.col_offset)
                if site in helper_names:
                    node.func = ast.copy_location(ast.Name(helper_names[site], ast.Load()), node.func)
                return node

            def visit_AnnAssign(self, node):
                self.generic_visit(node)
                if node.lineno in plan.local_lines:
                    return ast.copy_location(ast.Assign([node.target], node.value), node)
                return node

        tree = Rewrite().visit(tree)
        parameters = []
        for i, p in enumerate(original.signature.parameters.values()):
            parameters.append(p.replace(annotation=inspect.Parameter.empty) if i in plan.dynamic else p)
        for i in plan.dynamic:
            tree.body[0].args.args[i].annotation = None
        ast.fix_missing_locations(tree)
        src = ast.unparse(tree) + "\n"
        referenced = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        } - local_names
        scope = {name: value for name, value in scope.items() if name in referenced}
        obj = object.__new__(cls)
        obj.fn = types.FunctionType(original.fn.__code__, scope, original.fn.__name__, original.fn.__defaults__,
                                    original.fn.__closure__)
        obj.fn.__module__ = original.fn.__module__
        obj.fn.__qualname__ = original.fn.__qualname__
        obj.signature = original.signature.replace(parameters=parameters)
        obj.fn.__signature__ = obj.signature
        obj.raw_src = src.splitlines(keepends=True)
        obj.starting_line_number = original.starting_line_number
        obj._fn_name = original._fn_name
        obj._src = src
        obj.hash = None
        obj._hash_lock = threading.RLock()
        obj.used_global_vals = {}
        for name in ("__doc__", "__name__", "__qualname__", "__module__", "module", "version", "debug", "noinline"):
            setattr(obj, name, getattr(original, name))
        obj.__globals__ = scope
        obj._repr = None
        obj.launch_metadata = None
        obj.do_not_specialize = list(original.do_not_specialize) + list(plan.dynamic)
        obj.do_not_specialize_on_alignment = list(original.do_not_specialize_on_alignment) + list(plan.dynamic)
        obj.params = [
            KernelParam(i, p, original.params[i].do_not_specialize or i in plan.dynamic,
                        original.params[i].do_not_specialize_on_alignment or i in plan.dynamic)
            for i, p in enumerate(parameters)
        ]
        obj.arg_names = list(original.arg_names)
        obj.constexprs = [p.num for p in obj.params if p.is_constexpr]
        obj.pre_run_hooks = []
        obj.kernel = None
        obj.device_caches = defaultdict(obj.create_binder)
        obj.reuse_identity = digest(
            (src, plan.dynamic, plan.local_lines,
             tuple((site, scope[name].cache_key)
                   for site, name in helper_names.items()), config.RULE_VERSION, config.ABI_VERSION))
        return obj


def changed(plan):
    return bool(plan.dynamic or plan.local_lines or any(changed(child) for _, child in plan.helpers))
