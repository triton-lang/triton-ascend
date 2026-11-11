from collections import namedtuple
from types import SimpleNamespace

import pytest

from triton.backends.ascend import launcher

PREFIX = tuple(object() for _ in range(9))
Pair = namedtuple("Pair", ["pointer", "size"])


@pytest.mark.parametrize("signature,values,expected", [
    ({0: ()}, ((), ), ()),
    ({0: ("i32", )}, ((7, ), ), (7, )),
    ({0: ("i32", ("fp32", "constexpr")), 1: "i32"}, ((7, (2.5, None)), 9), (7, 2.5, None, 9)),
    ({0: "constexpr", 1: ("i32", "i32")}, ((8, 8), (2, 3)), ((8, 8), 2, 3)),
    ({0: ("*fp32", "i32")}, (Pair(1024, 7), ), (1024, 7)),
])
def test_launcher_flattens_tuple_values(signature, values, expected):
    launch = launcher.wrap_handle_tensordesc(lambda *args: args, signature)
    assert launch(*PREFIX, *values) == PREFIX + expected


def test_launcher_keeps_scalar_fast_path():
    launch = lambda *args: args
    assert launcher.wrap_handle_tensordesc(launch, {0: "*fp32", 1: "i32"}) is launch


@pytest.mark.parametrize("nested", [False, True])
def test_launcher_expands_descriptor_at_tuple_leaf(nested):
    descriptor = SimpleNamespace(base=1024, shape=[8, 16], strides=[16, 1], padding="nan")
    signature = ("i32", "tensordesc<fp32[8,16]>") if nested else "tensordesc<fp32[8,16]>"
    value = (7, descriptor) if nested else descriptor
    launch = launcher.wrap_handle_tensordesc(lambda *args: args, {0: signature, 1: "i32"})
    expected = (1024, 8, 16, 16, 1, True, 8, 16, 16, 1, 23)
    assert launch(*PREFIX, value, 23) == PREFIX + ((7, ) if nested else ()) + expected


def test_nested_descriptor_signature_matches_flat_signature():
    nested = {0: ("i32", ("tensordesc<fp32[8,16]>", "constexpr")), 1: "i32"}
    flat = {0: "i32", 1: "tensordesc<fp32[8,16]>", 2: "constexpr", 3: "i32"}
    assert launcher.argument_types(nested) == launcher.argument_types(flat)
