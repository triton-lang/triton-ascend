"""A5 FP32 batch helper accounting from the pinned RegBase Cube templates.

BatchL1Mmad emits one MMA per matrix, while ND2NZ and Fixpipe may transfer
several matrices with one descriptor. Descriptor field limits and L0 chunks
must therefore be considered separately from the batch dimension.
"""
from .backend_cost_model import Unsupported, known


def integers(values):
    return [int(known(v, 'unknown batch helper extent/control')) for v in values]


def batch_nd2nz(schedule, args):
    if len(args) != 23 or schedule.role != 'aic':
        raise Unsupported('unsupported batch ND2NZ ABI')
    batch, rows, cols = integers(args[3:6])
    stride, row_stride, inner = integers(args[6:9])
    layout = integers(args[12:17])
    dst_stride = integers(args[17:18])[0]
    if min(batch, rows, cols, *layout, stride, row_stride, dst_stride) <= 0 or inner != 1 or layout[0] != batch:
        raise Unsupported('unsupported batch ND2NZ layout')
    # compat/DMA/Cbuf/nd2nz.cpp: source stride is bounded in elements AND bytes.
    fits = all(x < 65536 for x in (batch, row_stride, stride, stride * 4, dst_stride, dst_stride // 8))
    commands = 1 if fits else batch * (1 if row_stride < 65536 else rows)
    source = batch * rows * cols * 4
    padded = 4
    for size in layout:
        padded *= size
    for _ in range(commands):
        schedule.op(
            4, 'gm_to_l1_batch_nd2nz', schedule.assumptions['dma_startup_cycles'] +
            max(source, padded) / commands / schedule.assumptions['dma_bytes_per_cycle'])
    schedule.dma_read_bytes += source
    schedule.counts.update(nd2nz_calls=commands, nd2nz_source_bytes=source, nd2nz_padded_bytes=padded,
                           batch_nd2nz_helpers=1)


def batch_mma(schedule, name, args):
    if len(args) != 52 or schedule.role != 'aic':
        raise Unsupported('unsupported batch MMA ABI')
    batch, batch_b, batch_c = integers([args[3], args[16], args[33]])
    m, k, n = integers(args[27:30])
    if batch < 2 or batch != batch_b or batch != batch_c or min(m, k, n) <= 0:
        raise Unsupported('unsupported batch MMA dimensions')
    # BatchL1Mmad uses full L0 buffers; this implementation admits only aligned
    # FP32 tiles. Unaligned layouts need independent layout/stride accounting.
    if m % 16 or k % 8 or n % 16:
        raise Unsupported('unaligned batch MMA tile')
    if args[47:50] != [-1, -1, -1] or args[50] != 0:
        raise Unsupported('batch MMA alternate/unit-flag protocol not implemented')
    per_chunk = min(64 * 1024 // (4 * m * k), 64 * 1024 // (4 * k * n), 256 * 1024 // (4 * m * n), batch)
    if per_chunk <= 0:
        raise Unsupported('batch MMA matrix exceeds one L0 buffer')
    wait_a, wait_b, release_a, release_b = integers(args[43:47])
    schedule.flag('llvm.hivm.WAIT.FLAG.IMM', [2, 3, 0])
    for event in (wait_a, wait_b):
        if event >= 0:
            schedule.flag('llvm.hivm.WAIT.FLAG.IMM', [4, 3, event])
    chunks = 0
    for start in range(0, batch, per_chunk):
        count = min(per_chunk, batch - start)
        chunks += 1
        for size in (count * m * k * 4, count * k * n * 4):
            schedule.op(
                3, 'l1_to_l0',
                schedule.assumptions['l1_l0_startup_cycles'] + size / schedule.assumptions['l1_l0_bytes_per_cycle'])
            schedule.counts['l1_to_l0_bytes'] += size
        schedule.flag('llvm.hivm.SET.FLAG.IMM', [3, 2, 0])
        schedule.flag('llvm.hivm.WAIT.FLAG.IMM', [3, 2, 0])
        for _ in range(count):
            schedule.op(
                2, name, schedule.assumptions['cube_startup_cycles'] +
                2 * m * k * n / schedule.assumptions['cube_flops_per_cycle'], (schedule.pipes[3], ))
            schedule.counts.update(mma_calls=1, useful_flops=2 * m * k * n, padded_flops=2 * m * k * n)
            schedule.counts['mma_shape:' + str((m, k, n))] += 1
        schedule.flag('llvm.hivm.SET.FLAG.IMM', [2, 3, 0])
        if start + count < batch:
            schedule.flag('llvm.hivm.WAIT.FLAG.IMM', [2, 3, 0])
    for event in (release_a, release_b):
        if event >= 0:
            schedule.flag('llvm.hivm.SET.FLAG.IMM', [3, 4, event])
    schedule.counts.update(batch_mma_helpers=1, batch_l0_chunks=chunks)


def batch_fixpipe(schedule, name, args):
    to_ub = name.endswith('_ubuf')
    if len(args) != (30 if to_ub else 29) or schedule.role != 'aic':
        raise Unsupported('unsupported batch Fixpipe ABI')
    if args[27] != 0:
        raise Unsupported('batch Fixpipe unit-flag protocol not implemented')
    batch, m, n = integers(args[16:19])
    source_stride, source_c0, dest_stride, row_stride = integers([args[8], args[11], args[19], args[20]])
    if min(batch, m, n, source_stride, source_c0, dest_stride, row_stride) <= 0:
        raise Unsupported('unsupported batch Fixpipe layout')
    if source_stride % source_c0:
        raise Unsupported('unaligned batch Fixpipe stride')
    fits = max(batch, source_stride // source_c0, dest_stride) <= 65535
    if to_ub and not fits:
        raise Unsupported('batch Fixpipe UB descriptor exceeds field limits')
    commands = 1 if fits else batch
    # NO_DUAL UB helper extends n to its row stride, as in the actual template.
    size = batch * m * (max(n, row_stride) if to_ub else n) * 4
    for _ in range(commands):
        schedule.op(
            10, 'fixpipe_to_ub' if to_ub else 'fixpipe_to_gm', schedule.assumptions['fixpipe_startup_cycles'] +
            size / commands / schedule.assumptions['fixpipe_bytes_per_cycle'])
    if not to_ub:
        schedule.dma_write_bytes += size
    schedule.counts.update(fixpipe_calls=commands, fixpipe_bytes=size, batch_fixpipe_helpers=1)
