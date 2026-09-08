"""Validate declarations before any builder call or attribute mutation."""
from types import SimpleNamespace

import pytest

from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.language.extra.cann.extension.custom_op import _add_optional_flatten_symbols_attr

pytestmark = pytest.mark.backend("none")


class RejectBuilderCalls:

    def __getattr__(self, name):
        raise AssertionError(f"Invalid declaration reached builder.{name}")


@pytest.mark.parametrize("table,error", [
    (None, TypeError),
    ([], TypeError),
    ([(1, "f")], TypeError),
    ({}, ValueError),
    ({True: "f"}, TypeError),
    ({False: "f"}, TypeError),
    ({1.5: "f"}, TypeError),
    ({1.0: "f"}, TypeError),
    ({"1": "f"}, TypeError),
    ({1: "first", "1": "second"}, TypeError),
    ({1.1: "first", 1.9: "second"}, TypeError),
    ({0: "f"}, ValueError),
    ({-1: "f"}, ValueError),
    ({1 << 63: "f"}, ValueError),
    ({1: None}, TypeError),
    ({1: 7}, TypeError),
    ({1: []}, TypeError),
    ({1: ""}, ValueError),
    ({1: "valid", 2: None}, TypeError),
])
def test_invalid_flatten_declaration_is_atomic(table, error):
    attrs = {"preserved": object()}
    before = dict(attrs)
    with pytest.raises(error, match="flatten_symbols"):
        _add_optional_flatten_symbols_attr(SimpleNamespace(flatten_symbols=table), RejectBuilderCalls(), attrs)
    assert attrs == before


def test_absent_flatten_declaration_does_not_use_builder():
    attrs = {"preserved": object()}
    before = dict(attrs)
    _add_optional_flatten_symbols_attr(SimpleNamespace(), RejectBuilderCalls(), attrs)
    assert attrs == before


def _print_module_attrs(builder, attrs):
    module = builder.create_module()
    for name, attr in attrs.items():
        module.set_attr(name, attr)
    return str(module)


def test_flatten_declaration_with_real_builder():
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    builder = ascend_ir.ascendnpu_ir_builder(context, "Ascend910B1")
    attrs = {}
    table = {2: "second", 1: "first"}
    _add_optional_flatten_symbols_attr(SimpleNamespace(flatten_symbols=table), builder, attrs)
    text = _print_module_attrs(builder, attrs)
    assert "flatten_ranks = [1, 2]" in text
    assert 'flatten_symbols = ["first", "second"]' in text
    assert list(table.items()) == [(2, "second"), (1, "first")]


def test_flatten_symbol_escaping_with_real_builder():
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    builder = ascend_ir.ascendnpu_ir_builder(context, "Ascend910B1")
    attrs = {}
    symbol = 'quoted"symbol\\suffix'
    _add_optional_flatten_symbols_attr(SimpleNamespace(flatten_symbols={1: symbol}), builder, attrs)
    text = _print_module_attrs(builder, attrs)
    # Inspect the real MLIR printer's quote and backslash escaping.
    assert r'flatten_symbols = ["quoted\22symbol\\suffix"]' in text
