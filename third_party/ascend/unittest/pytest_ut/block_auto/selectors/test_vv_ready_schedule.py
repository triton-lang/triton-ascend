import copy
import json
from pathlib import Path
import struct

import pytest

import backend_cost_model as backend
from vv_ready_schedule import ReadyVectorSchedule
from vv_skill_model import InstructionModel
from vv_optimized_ir import executable_sections, vector_bodies, verified_vectors


def scheduler(window=64):
    return ReadyVectorSchedule({}, dict(backend.ASSUMPTIONS, issue_queue_length=window), InstructionModel())


def test_waiting_consumer_does_not_block_independent_issue():
    s = scheduler()
    producer = s.node('', '%a', 'load', 'load', 1, 20, 2)
    dependent = s.node('%a', '%b', 'alu', 'alu', 1, 4, 2)
    independent = s.node('', '%c', 'other', 'other', 1, 4, 2)
    s.finalize()
    assert s.scheduled[independent][0] < s.scheduled[producer][1]
    assert s.scheduled[dependent][0] >= s.scheduled[producer][1]


def test_queue_retirement_waits_for_all_older_instructions():
    s = scheduler(2)
    oldest = s.node('', '%slow', 'load', 'load', 1, 100, 1)
    fast = s.node('', '%fast', 'other', 'other', 1, 1, 1)
    after_window = s.node('', '%third', 'third', 'third', 1, 1, 1)
    fourth = s.node('', '%fourth', 'fourth', 'fourth', 1, 1, 1)
    s.finalize()
    assert s.scheduled[fast][1] < s.scheduled[oldest][1]
    assert s.scheduled[after_window][0] >= s.scheduled[oldest][1]
    assert s.scheduled[fourth][0] >= s.scheduled[oldest][1]


def test_single_and_dual_issue_share_existing_reservations():
    s = scheduler()
    a = s.node('', '%a', 'same', 'same', 20, 20, 1)
    b = s.node('', '%b', 'same', 'same', 20, 20, 2)
    c = s.node('', '%c', 'same', 'same', 1, 1, 1)
    s.finalize()
    assert s.scheduled[b][0] < s.scheduled[a][1]
    assert s.scheduled[c][0] >= s.scheduled[a][1]


def test_control_branch_and_scalar_dependencies_are_not_free():
    s = scheduler()
    s.control('add i32 1, 2', '%a')
    s.control('icmp eq i32 %a, 3', '%condition')
    s.control('br i1 %condition, label %yes, label %no', None)
    control = next(iter(s.control_ref))
    work = s.node('', '%value', 'load', 'load', 1, 4, 2)
    s.finalize()
    assert s.scheduled[work][0] >= s.scheduled[control][1]
    assert s.counts['vf_control:br'] == 1


def test_phi_aliases_preserve_loop_carried_vector_dependency():
    code = '''define internal ptc_simdvf void @vf() {
  br label %loop
loop:
  %i = phi i32 [ 0, %0 ], [ %next, %loop ]
  %a = phi <64 x float> [ undef, %0 ], [ %b, %loop ]
  %b = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %a, <64 x float> undef, <256 x i1> undef)
  %next = add i32 %i, 1
  %more = icmp slt i32 %next, 3
  br i1 %more, label %loop, label %exit
exit:
  ret void
}'''
    made = []

    def factory(table, assumptions):
        s = ReadyVectorSchedule(table, assumptions, InstructionModel())
        made.append(s)
        return s

    result = backend.Analyzer(code, {}, vector_factory=factory).trace('vf', [], 0)
    assert result['counts']['llvm.hivm.vmul.s.x.v64f32'] == 3
    s = made[0]
    multiply = [i for i, n in enumerate(s.nodes) if n['unit'] == 'vector_alu']
    assert len(multiply) == 3
    for a, b in zip(multiply, multiply[1:]):
        assert s.scheduled[b][0] >= s.scheduled[a][1]


def minimal_elf(path, code=b'abcd', note=b'not executable'):
    names = b'\0.text\0.comment\0.shstrtab\0'
    data = bytearray(64)
    data[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<H', data, 18, 900)
    regions = []
    for payload in (code, note, names):
        regions.append((len(data), len(payload)))
        data.extend(payload)
    shoff = len(data)
    struct.pack_into('<Q', data, 40, shoff)
    struct.pack_into('<HHH', data, 58, 64, 4, 3)
    data.extend(bytes(64))
    for name, flags, region in zip((1, 7, 16), (6, 48, 0), regions):
        data.extend(struct.pack('<IIQQQQIIQQ', name, 1 if name != 16 else 3, flags, 0, *region, 0, 0, 4, 0))
    path.write_bytes(data)


def test_binary_proof_ignores_comments_but_rejects_changed_code(tmp_path):
    a, b = tmp_path / 'a.o', tmp_path / 'b.o'
    minimal_elf(a)
    minimal_elf(b, note=b'different path and compiler comment')
    assert executable_sections(a) == executable_sections(b)
    minimal_elf(b, code=b'abce')
    assert executable_sections(a) != executable_sections(b)
    b.write_bytes(b'not ELF')
    with pytest.raises(ValueError):
        executable_sections(b)


def test_replace_only_vector_bodies_and_reject_dropped_abi():
    raw = '''define void @kernel() {\n  call ptc_simdvf void @vf(i32 2)\n  ret void\n}
define internal ptc_simdvf void @vf(i32 %a) {\n  ret void\n}'''
    optimized = 'define internal ptc_simdvf void @vf(i32 %a) {\n  %b = add i32 %a, 1\n  ret void\n}'
    combined = vector_bodies(raw, optimized)
    assert combined.startswith(raw.split('define internal')[0])
    assert '%b = add' in combined
    with pytest.raises(backend.Unsupported, match='ABI'):
        vector_bodies(raw, optimized.replace('i32 %a)', 'i64 %a)'))
    with pytest.raises(backend.Unsupported, match='missing'):
        vector_bodies(raw, '')


def test_optimized_evidence_is_required():
    with pytest.raises(backend.Unsupported, match='missing optimized'):
        verified_vectors('', {})


def removed_argument_fixture():
    raw = '''define void @kernel() {
  call ptc_simdvf void @vf(i32 9, i32 64)
  ret void
}
define internal ptc_simdvf void @vf(i32 %0, i32 %1) {
  ret void
}'''
    optimized = '''define void @kernel() {
  tail call ptc_simdvf void @vf(i32 noundef 64)
  ret void
}
define internal ptc_simdvf void @vf(i32 %0) {
  %2 = add i32 %0, 1
  ret void
}'''
    return raw, optimized


def test_proven_dead_argument_retains_call_binding_and_entry_label():
    raw, optimized = removed_argument_fixture()
    combined = vector_bodies(raw, optimized)
    assert 'call ptc_simdvf void @vf(i32 9, i32 64)' in combined
    vf = backend.parse_module(combined)['vf']
    assert vf.arguments == ['%optimized_unused_0', '%0']
    assert vf.entry == '1'
    assert '%2 = add i32 %0, 1' in combined


@pytest.mark.parametrize('change', ['ambiguous', 'different', 'dynamic', 'caller', 'multiple'])
def test_unproven_dead_argument_mapping_is_rejected(change):
    raw, optimized = removed_argument_fixture()
    if change == 'ambiguous':
        raw = raw.replace('i32 9, i32 64', 'i32 64, i32 64')
    elif change == 'different':
        optimized = optimized.replace('noundef 64', 'noundef 65')
    elif change == 'dynamic':
        raw = raw.replace('i32 9, i32 64', 'i32 9, i32 %x')
        optimized = optimized.replace('noundef 64', 'noundef %x')
    elif change == 'caller':
        optimized = optimized.replace('@kernel()', '@different()')
    elif change == 'multiple':
        optimized = optimized.replace('  tail call', '  call ptc_simdvf void @vf(i32 64)\n  tail call')
    with pytest.raises(backend.Unsupported, match='SIMD'):
        vector_bodies(raw, optimized)


@pytest.mark.parametrize('window', [0, -1, True, 1.5])
def test_invalid_queue_rejected(window):
    with pytest.raises(backend.Unsupported, match='queue length'):
        scheduler(window)
