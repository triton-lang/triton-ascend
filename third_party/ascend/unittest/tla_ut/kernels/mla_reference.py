"""FP32 reference and inputs for paged MLA examples."""

import torch


def make_inputs(lengths, family, seed):
    """Use random physical pages, including unused slots in shorter requests."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    page_counts = [(length + 127) // 128 for length in lengths]
    ids = torch.randperm(sum(page_counts), generator=generator, dtype=torch.int32)
    table = torch.full((len(lengths), max(page_counts)), -1, dtype=torch.int32)
    for row, pages in enumerate(ids.split(page_counts)):
        table[row, :pages.numel()] = pages
    device = torch.device('npu', torch.npu.current_device())

    def random(shape, query=False):
        if query and family == 'uniform':
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        data = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
        if query and family == 'peaked':
            data.mul_(8)
        return data.to(device)

    # Retain the input generation order of the pre-cleanup benchmark.
    q = random((len(lengths), 4, 24, 512), query=True)
    qr = random((len(lengths), 4, 24, 64), query=True)
    kv = random((ids.numel(), 128, 512))
    kr = random((ids.numel(), 128, 64))
    packed_q = torch.cat((q, qr), dim=-1)
    packed_kv = torch.cat((kv, kr), dim=-1)
    return packed_q, packed_kv, table.to(device), torch.tensor(lengths, dtype=torch.int32, device=device)


def reference(inputs):
    """Independent FP32 attention with one full softmax per query row."""
    q, kv, table, lengths = inputs
    batch, sq, heads, _ = q.shape
    table_cpu, lengths_cpu = table.cpu(), lengths.cpu().tolist()
    outputs, lses = [], []
    for request, length in enumerate(lengths_cpu):
        pages = table_cpu[request, :(length + 127) // 128].to(q.device, torch.int64)
        keys = kv.index_select(0, pages).reshape(-1, 576)[:length].float()
        query = q[request].reshape(-1, 576).float()
        positions = torch.arange(length, device=q.device)
        visible_end = length - sq + torch.arange(sq, device=q.device).repeat_interleave(heads)
        values, normalizers = [], []
        for start in range(0, sq * heads, 16):
            rows = slice(start, start + 16)
            logits = (query[rows, :512] @ keys[:, :512].T + query[rows, 512:] @ keys[:, 512:].T) * 0.04
            logits.masked_fill_(positions[None, :] > visible_end[rows, None], -float('inf'))
            values.append(logits.softmax(-1) @ keys[:, :512])
            normalizers.append(logits.logsumexp(-1))
        outputs.append(torch.cat(values).reshape(sq, heads, 512))
        lses.append(torch.cat(normalizers).reshape(sq, heads))
    return torch.stack(outputs), torch.stack(lses)


def precision(call, expected):
    actual = call()
    torch.npu.synchronize()
    assert isinstance(actual, tuple) and len(actual) == 2, 'Expected both output and LSE'
    result = {}
    for name, output, target, dtype, tolerance in zip(('output', 'lse'), actual, expected,
                                                      (torch.bfloat16, torch.float32), (0.005, 0.0001)):
        assert output.shape == target.shape and output.dtype == dtype, name
        assert torch.isfinite(output).all(), name
        error = output.float() - target
        maximum = (error.abs().amax() / target.abs().amax().clamp_min(1e-30)).item()
        rms = (error.square().mean().sqrt() / target.square().mean().sqrt().clamp_min(1e-30)).item()
        assert maximum <= tolerance and rms <= tolerance, (name, maximum, rms)
        result[name] = dict(normalized_max_error=maximum, relative_rms=rms)
    return result
