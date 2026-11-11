"""Static row-window diagnostics must precede emission of an ND2NZ op."""
import importlib
from pathlib import Path
import sys
import types

import pytest
import triton.language as tl

SOURCE = Path(__file__).resolve().parents[3] / 'language' / 'tla'
package = types.ModuleType('_nd2nz_contract_tla')
package.__path__ = [str(SOURCE)]
sys.modules.setdefault(package.__name__, package)
dma = importlib.import_module(package.__name__ + '.dma')
core = importlib.import_module(package.__name__ + '.core')


class Recorder:
    def __init__(self):
        self.calls = []

    def create_tla_nd2nz(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class Semantic:
    def __init__(self):
        self.builder = Recorder()

    def to_tensor(self, value):
        return tl.tensor(value, tl.int32 if type(value) is int else tl.float32)


def buffers():
    src = core.buffer(object(), core.buffer_type((128, 64), tl.bfloat16, 'gm'))
    dst = core.buffer(object(), core.buffer_type((4, 7, 16, 16), tl.bfloat16, 'l1'))
    return src, dst


def test_legacy_load_keeps_its_three_argument_builder_call():
    src, dst = buffers()
    src = core.buffer(object(), core.buffer_type((112, 64), tl.bfloat16, 'gm'))
    sem = Semantic()
    dma.load_gm_to_l1(src, dst, _semantic=sem)
    assert sem.builder.calls == [((src.handle, dst.handle, {}), {})]


def test_valid_static_window_is_forwarded():
    src, dst = buffers()
    sem = Semantic()
    dma.load_gm_to_l1(src, dst, dma.LoadGM2L1Params(row_count=48, source_row=80, destination_row=64),
                      _semantic=sem)
    assert sem.builder.calls[0][1] == dict(row_count=48, source_row=80, destination_row=64)


@pytest.mark.parametrize('fields', (
    dict(source_row=16), dict(destination_row=16), dict(row_count=0), dict(row_count=-16),
    dict(row_count=17), dict(row_count=16, source_row=1), dict(row_count=16, destination_row=-16),
    dict(row_count=48, source_row=96), dict(row_count=48, destination_row=80),
))
def test_invalid_static_window_is_rejected_before_emission(fields):
    sem = Semantic()
    with pytest.raises(ValueError, match='ND2NZ'):
        dma.load_gm_to_l1(*buffers(), dma.LoadGM2L1Params(**fields), _semantic=sem)
    assert not sem.builder.calls
