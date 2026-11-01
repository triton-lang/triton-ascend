"""Typed TLA source extensions for existing AVE/HIVM operations.

The installed wheel lacks the two pybind entry points. MMAD uses private TTIR
functions removed by the normal inliner. ExpSub splices a typed native AVE
region inside a constant condition removed by the normal canonicalizer, since
the installed AVE dialect has no inliner interface. Final IR is audited for
native operations without helper calls. No runtime preprocessing, binary
patching, changed compiler options, or numerical substitutes are used.
"""
import hashlib
from pathlib import Path
import tempfile
import triton.language as tl
from triton.language.core import builtin, _unwrap_if_constexpr
from triton.language.extra.tla.core import vector
from triton.language.extra.tla.cube import _logical_shape
from triton.language.extra.tla.simd import _all_mask, _scalar_handle


def install():
    from triton.compiler.code_generator import CodeGenerator
    if getattr(CodeGenerator, '_issue104_native_ext', False):
        return
    original = CodeGenerator.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.builder._tla_extension_module = self.module

    CodeGenerator.__init__ = init
    CodeGenerator._issue104_native_ext = True


def _call(semantic, arguments, body, returns=()):
    from triton._C.libtriton import ir
    builder = semantic.builder
    module = builder._tla_extension_module
    types = [str(a.get_type()) for a in arguments]
    signature = ', '.join(f'%a{i}: {ty}' for i, ty in enumerate(types))
    result = ' -> (' + ', '.join(returns) + ')' if returns else ''
    key = hashlib.sha256((signature + result + body).encode()).hexdigest()[:20]
    name = '__tla_native_' + key
    if not module.has_function(name):
        text = f'module {{ tt.func private @{name}({signature}){result} {{\n{body}\n}} }}'
        with tempfile.TemporaryDirectory(prefix='tla-native-op-') as directory:
            path = Path(directory) / 'op.mlir'
            path.write_text(text)
            helper = ir.parse_mlir_module(str(path), arguments[0].get_context())
        # ModuleOp.push_back requires an already-detached operation. Splice
        # the helper module's zero-argument body instead, preserving ownership.
        source_block = helper.get_function(name).get_operation().get_block()
        destination_block = module.get_function(module.get_entry_func_name()).get_operation().get_block()
        source_block.merge_block_before(destination_block)
    return builder.call(module.get_function(name), arguments)


@builtin
def vmuls(src, scalar, _semantic=None):
    """Native FP32 vector-scalar multiply through the existing TLA binding."""
    if src.dtype != tl.float32 or src.shape != (64, ):
        raise ValueError('vmuls currently requires vector<64xf32>')
    handle = _scalar_handle(src, scalar, _semantic)
    mask = _all_mask(64, _semantic)
    result = _semantic.builder.create_simd_vec_scalar('muls', src.handle, handle, mask.handle)
    return vector(result, src.type)


@builtin
def vexpdif(lhs, rhs, _semantic=None):
    if lhs.type != rhs.type or lhs.dtype != tl.float32 or lhs.shape != (64, ):
        raise ValueError('vexpdif currently requires two vector<64xf32> operands')
    mask = _all_mask(64, _semantic)
    ty, mt = str(lhs.handle.get_type()), str(mask.handle.get_type())
    # The installed AVE dialect has no inliner interface, unlike HIVM.
    # Splice a typed native-op region at this insertion point. The constant
    # condition is removed by the normal TTIR canonicalizer; final IR must
    # contain vexpdif directly, with no helper call or conditional remaining.
    from triton._C.libtriton import ir
    builder = _semantic.builder
    text = f'''module {{ tt.func private @primitive(%a: {ty}, %b: {ty}, %p: {mt}) -> {ty} {{
      %r = scf.execute_region -> {ty} {{
        %e = ave.hir.vexpdif %a, %b, %p : {ty}, {mt}
        scf.yield %e : {ty}
      }}
      tt.return %r : {ty}
    }} }}'''
    with tempfile.TemporaryDirectory(prefix='tla-native-exp-') as directory:
        path = Path(directory) / 'op.mlir'
        path.write_text(text)
        helper = ir.parse_mlir_module(str(path), lhs.handle.get_context())
    fn = helper.get_function('primitive')
    for i, handle in enumerate((lhs.handle, rhs.handle, mask.handle)):
        fn.args(i).replace_all_uses_with(handle)
    native = []
    helper.walk(lambda op: native.append(op) if op.get_name() == 'ave.hir.vexpdif' else None)
    condition = builder.get_int1(True)
    branch = builder.create_if_op([lhs.handle.get_type()], condition, True)
    native[0].get_block().merge_block_before(branch.get_then_block())
    point = builder.get_insertion_point()
    builder.set_insertion_point_to_end(branch.get_else_block())
    builder.create_yield_op([lhs.handle])
    builder.restore_insertion_point(point)
    return vector(branch.get_result(0), lhs.type)


@builtin
def matmul(a, b, acc, transpose_b=False, clear_accum=False, transpose_a=False, _semantic=None):
    ta, tb, clear = (bool(_unwrap_if_constexpr(x)) for x in (transpose_a, transpose_b, clear_accum))
    m, k = _logical_shape(a.shape)
    k2, n = _logical_shape(b.shape)
    if ta:
        m, k = k, m
    if tb:
        k2, n = n, k2
    if (a.space, b.space, acc.space) != ('l1', 'l1', 'l0c'):
        raise ValueError('MMAD requires L1/L1/L0C')
    if a.dtype != b.dtype or k != k2 or _logical_shape(acc.shape) != (m, n):
        raise ValueError('MMAD logical shapes or dtypes do not agree')
    attrs = ', '.join(x for x, enabled in [('a_transpose', ta), ('b_transpose', tb)] if enabled)
    attrs = ' {' + attrs + '}' if attrs else ''
    at, bt, ct = (str(x.handle.get_type()) for x in (a, b, acc))
    body = f'''%init = arith.constant {str(clear).lower()}
%m = arith.constant {m} : index
%k = arith.constant {k} : index
%n = arith.constant {n} : index
hivm.hir.mmadL1{attrs} ins(%a0, %a1, %init, %m, %k, %n : {at}, {bt}, i1, index, index, index) outs(%a2 : {ct})
tt.return'''
    _call(_semantic, [a.handle, b.handle, acc.handle], body)


install()
