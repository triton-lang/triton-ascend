"""Host checks for NZ16 to NZ32 byte ownership and packing."""
import pytest


@pytest.mark.parametrize("rows", [32, 64, 128])
def test_f16_n512_cce_offsets_and_pack(rows):
    # Simulate physical byte ownership, including all 8-row store groups.
    written = set()
    for r in range(0, rows, 8):
        for k in range(16):
            reads = {2 * ((2 * k + j) * 16 * rows + (r // 16) * 256
                         + (r % 16) * 16 + t) + b
                     for j in range(2) for t in range(128) for b in range(2)}
            assert not reads & written
            store = 64 * rows * k + 512 * (r // 16) + 32 * (r % 16)
            output = set(range(store, store + 256))
            assert output <= reads
            assert not output & written
            assert max(output) < rows * 512 * 2
            written |= output
    assert len(written) == rows * 512
    # CCE ZERO/TWO carry even/odd columns of first NZ16 block;
    # ONE/THREE carry even/odd columns of the second NZ16 block.
    packed = [None] * 256
    for row in range(8):
        for col in range(16):
            lane = row * 8 + col // 2
            packed[4 * lane + 2 * (col % 2)] = (row, col)
            packed[4 * lane + 2 * (col % 2) + 1] = (row, col + 16)
    lut = [(i // 32) * 32 + (i % 16) * 2 + (i % 32) // 16 for i in range(256)]
    assert [packed[i] for i in lut] == [(r, c) for r in range(8) for c in range(32)]


def test_f16_n512_explicit_symbol():
    from importlib import import_module
    from triton._C.libtriton import ir
    from triton._C.libtriton.ascend import ir as ascend_ir

    custom_op = import_module("triton.language.extra.cann.extension.custom_op")
    with ir.context() as context:
        ir.load_dialects(context)
        ascend_ir.load_dialects(context)
        builder = ascend_ir.ascendnpu_ir_builder(context)
        op = custom_op._get_op_class("__builtin_online_softmax_nz")
        attrs = custom_op._make_attrs(op, builder, is_macro=False)
        module = builder.create_module()
        module.set_attr("symbol", attrs["symbol"])
        assert module.get_operation().get_str_attr("symbol") == "__builtin_online_softmax_nz"
