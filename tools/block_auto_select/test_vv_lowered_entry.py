"""Check optimized ABI handling and independent compiler-produced DMA words."""
import json
from pathlib import Path

import pytest

from backend_cost_model import Analyzer, ASSUMPTIONS, Unsupported
from vv_lowered_entry import LoweredEntrySchedule, decode_dma, lowered_intrinsic, validate_entry
from vv_simt_helpers import VerifiedSimtHelpers

FIXTURES = json.loads(Path(__file__).with_name('lowered_dma_cases.json').read_text())['rows']
READ = 'llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV'
ND = 'llvm.hivm.ND.DMA.OUT.TO.UB.b32'


@pytest.mark.parametrize('record', FIXTURES, ids=lambda r: r['name'])
def test_compiler_produced_dma_control_words(record):
    assert decode_dma(record['intrinsic'], [None, None, *record['words']]) == (record['direction'], record['bytes'])


@pytest.mark.parametrize('args', [
    [None, None, None, 0],
    [None, None, -1, 0],
    [None, None, 2**64, 0],
    [None, None, 16, 0],
    [None, None, 32 << 25, 0],
    [None, None, (32 << 25) | 16 | 1, 0],  # unsupported stream id
    [None, None, (32 << 25) | 16 | (1 << 16), 0],  # padding/control field
    [None, None, True, 0],
])
def test_invalid_or_unresolved_dma_never_becomes_zero_work(args):
    with pytest.raises(Unsupported):
        decode_dma(READ, args)


@pytest.mark.parametrize('second', [1 | (1 << 20) | (1 << 40), 1 | (1 << 20) | (1 << 48), 1 | (1 << 20) | (1 << 60)])
def test_nd_padding_and_unverified_cache_modes_rejected(second):
    with pytest.raises(Unsupported):
        decode_dma(ND, [None, None, 16 | (1 << 24) | (1 << 44), second])


def test_nd_requires_observed_zero_padding():
    s = LoweredEntrySchedule(ASSUMPTIONS, None)
    args = [None, None, 16 | (1 << 24) | (1 << 44), 1 | (1 << 20)]
    with pytest.raises(Unsupported, match='padding'):
        lowered_intrinsic(s, ND, args)
    assert lowered_intrinsic(s, 'llvm.hivm.SET.PAD.CNT.NDDMA', [0])
    assert lowered_intrinsic(s, ND, args)
    assert s.result()['dma_read_bytes'] == 4
    with pytest.raises(Unsupported):
        lowered_intrinsic(s, 'llvm.hivm.SET.PAD.CNT.NDDMA', [1])


def analyzer(text):
    return Analyzer(text, {}, lowered_entry=True, kernel_factory=lambda c: LoweredEntrySchedule(c, None))


def test_nested_optimized_helper_keeps_shared_dma_flag_dependencies():
    text = '''
define void @main() {
entry:
  call fastcc void @load_gm_to_ubuf_1d_float() #1
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  ret void
}
define private fastcc void @load_gm_to_ubuf_1d_float() {
entry:
  call void @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV(ptr addrspace(6) null, ptr addrspace(1) null, i64 1073741840, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  ret void
}
'''
    result = analyzer(text).trace('main', [], 0)
    assert result['dma_read_bytes'] == 32
    assert result['counts']['load_dma_calls'] == 1
    assert sum(e['kind'] == 'end' for e in result['events']) == 1
    assert result['counts']['flags'] == 2


def test_ub_pointer_offsets_can_resolve_alignment_control():
    text = '''
define void @main() {
entry:
  %p = getelementptr bfloat, ptr addrspace(6) inttoptr (i64 64 to ptr addrspace(6)), i64 16
  %v = ptrtoint ptr addrspace(6) %p to i64
  %ok = icmp eq i64 %v, 96
  br i1 %ok, label %good, label %bad
good:
  ret void
bad:
  call void @unknown()
  ret void
}
'''
    assert analyzer(text).trace('main', [], 0)['block_visits']['good'] == 1


@pytest.mark.parametrize('value', [
    'call i64 @llvm.hivm.GET.SYS.VA.BASE()',
    'call i64 @llvm.hivm.GET.CTRL()',
    'ptrtoint ptr addrspace(1) null to i64',
    'load i64, ptr addrspace(6) null',
])
def test_runtime_addresses_and_tensor_data_stay_unknown(value):
    text = '''
define void @main() {
entry:
  %v = VALUE
  %ok = icmp eq i64 %v, 0
  br i1 %ok, label %good, label %bad
good:
  ret void
bad:
  ret void
}
'''.replace('VALUE', value)
    with pytest.raises(Unsupported, match='data-dependent branch'):
        analyzer(text).trace('main', [], 0)


def test_entry_abi_must_match_even_when_internal_helper_abi_may_change():
    raw = 'define void @main(ptr addrspace(1) %p, i32 %n) {\nentry:\n  ret void\n}'
    validate_entry(raw, raw.replace('i32 %n', 'i32 noundef %n'), 'main')
    with pytest.raises(Unsupported, match='ABI changed'):
        validate_entry(raw, raw.replace(', i32 %n', ''), 'main')
    with pytest.raises(Unsupported, match='missing'):
        validate_entry(raw, raw, 'other')


def test_unknown_intrinsic_is_not_free():
    with pytest.raises(Unsupported, match='unmodelled call'):
        analyzer('define void @main() {\nentry:\n  call void @llvm.hivm.NEW.DMA()\n  ret void\n}').trace('main', [], 0)


def test_scalar_stack_arguments_and_writes_propagate_across_vector_calls():
    text = '''
define void @main() {
entry:
  %p = alloca i32
  store i32 7, ptr %p
  call ptc_simdvf void @vf(ptr nonnull %p)
  call ptc_simdvf void @vf(ptr nonnull %p)
  %n = load i32, ptr %p
  %ok = icmp eq i32 %n, 5
  br i1 %ok, label %good, label %bad
good:
  ret void
bad:
  call void @wrong_stack_effect()
  ret void
}
define internal ptc_simdvf void @vf(ptr %arg) {
entry:
  %p = alloca i32
  store i32 999, ptr %p
  %n = load i32, ptr %arg
  %next = sub i32 %n, 1
  store i32 %next, ptr %arg
  ret void
}
'''
    a = analyzer(text)
    assert a.trace('main', [], 0)['block_visits']['good'] == 1
    assert len(a.vf_cache) == 2
    # Reusing the cached vector profiles must also replay scalar stack effects.
    assert a.trace('main', [], 0)['block_visits']['good'] == 1
    assert len(a.vf_cache) == 2


def test_optimized_simt_wrapper_binds_shape_and_cache_does_not_mix_shapes():
    name = 'indirect_load_nonvolatile_1d_bfloat16_t_int64_t'
    text = '''
define void @HELPER(ptr addrspace(1) %p, i64 %size) {
entry:
  %n = trunc i64 %size to i32
  call void @llvm.hivm.store.vfsimt.info(i64 4295032834)
  call simt_entry void @body(ptr addrspace(1) %p, i32 %n)
  ret void
}
define simt_entry void @body(ptr addrspace(1) %p, i32 %n) {
entry:
  %tid = call i32 @llvm.hivm.get.TID.X()
  %active = icmp ult i32 %tid, %n
  br i1 %active, label %copy, label %done
copy:
  %x = load bfloat, ptr addrspace(1) %p
  store bfloat %x, ptr addrspace(6) null
  br label %done
done:
  ret void
}
'''.replace('HELPER', name)
    m = VerifiedSimtHelpers(text, {},
                            dict(simt_transaction_bytes=64, simt_transition_cycles=32, simt_instruction_cycles=1),
                            lowered_abi=True)
    assert m.profile(name, [None, 1])['counts']['gm_load_bytes'] == 2
    assert m.profile(name, [None, 2])['counts']['gm_load_bytes'] == 4
    assert m.profile(name, [None, 1])['counts']['gm_load_bytes'] == 2
    assert len(m.cache) == 2
    with pytest.raises(Unsupported, match='dynamic lowered SIMT scalar'):
        m.profile(name, [None, None])
