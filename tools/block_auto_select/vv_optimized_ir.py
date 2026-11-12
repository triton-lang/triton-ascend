"""Bind optimized SIMD bodies to the candidate's executable sections.

Preparation reruns the CPU compiler with a recorded profile, then checks every
loadable section and relocation section against the candidate. No candidate
binary is replaced. Analysis only reads the sealed outputs; it cannot compile.
"""
import hashlib
import json
from pathlib import Path
import re
import struct

from backend_cost_model import Unsupported, parse_module, split_top, type_and_value
from cpu_resource_gate import sha256

VERSION = 'optimized-vf-ready-queue-v1'


def executable_sections(path):
    data = Path(path).read_bytes()
    if len(data) < 64 or data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected a little-endian ELF64 binary')
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, strings = struct.unpack_from('<HHH', data, 58)
    if size != 64 or not 0 < count < 65535 or strings >= count or offset + count * size > len(data):
        raise ValueError('invalid ELF section table')
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * size) for i in range(count)]
    names_section = sections[strings]
    names = data[names_section[4]:names_section[4] + names_section[5]]
    result = {
        'elf_machine': struct.unpack_from('<H', data, 18)[0], 'elf_flags': struct.unpack_from('<I', data, 48)[0],
        'sections': {}
    }
    for section in sections:
        start, kind, flags, address, at, length, link, info, align, entsize = section
        if not (flags & 2 or kind in (4, 9)):
            continue
        if start >= len(names) or kind != 8 and at + length > len(data):
            raise ValueError('invalid executable section range')
        name = names[start:].split(b'\0', 1)[0].decode()
        if name in result['sections']:
            raise ValueError('duplicate executable section')
        payload = b'' if kind == 8 else data[at:at + length]
        result['sections'][name] = dict(kind=kind, flags=flags, address=address, size=length, alignment=align,
                                        sha256=hashlib.sha256(payload).hexdigest())
    if '.text' not in result['sections']:
        raise ValueError('missing executable text')
    return result


def record(path):
    return dict(path=str(path), sha256=sha256(path))


def restore_removed_arguments(name, before, after, raw, optimized):
    """Retain the original call ABI after proven dead-argument elimination.

    Only constant call operands with an unambiguous mapping in every caller
    are supported. Dynamic/duplicate operands and changed callers fail closed.
    This edits analysis text only; executable-section verification is unchanged.
    """
    signature = r'ptc_simdvf void @[^ (]+\((.*)\)'
    old_header, new_header = before.splitlines()[0], after.splitlines()[0]
    old_args = split_top(re.search(signature, old_header)[1])
    new_match = re.search(signature, new_header)
    new_args = [a for a in split_top(new_match[1]) if a]
    if not len(new_args) < len(old_args):
        raise Unsupported('optimizer changed SIMD argument ABI: ' + name)

    def constant(arg):
        dtype, payload = type_and_value(arg)
        payload = re.sub(
            r'^(?:(?:nonnull|noundef|nocapture|noalias|readonly|writeonly)\s+|'
            r'align \d+\s+|dereferenceable\(\d+\)\s+)*', '', payload)
        if not re.fullmatch(r'(?:null|-?\d+|inttoptr \(i\d+ -?\d+ to ptr(?: addrspace\(\d+\))?\))', payload):
            raise Unsupported('cannot prove optimized SIMD argument mapping: ' + name)
        return dtype, payload

    def sites(text, arity):
        result = {}
        for parent, function in parse_module(text).items():
            for lines in function.blocks.values():
                for line in lines:
                    if not re.search(r'\bcall\b[^\n]*@' + re.escape(name) + r'\(', line):
                        continue
                    match = re.search(r'\bcall ptc_simdvf void @' + re.escape(name) + r'\((.*)\)(?:\s+#[0-9]+)?$', line)
                    if not match or parent in result:
                        raise Unsupported('ambiguous optimized SIMD call site: ' + name)
                    args = [constant(a) for a in split_top(match[1]) if a]
                    if len(args) != arity:
                        raise Unsupported('optimized SIMD call arity differs: ' + name)
                    result[parent] = args
        return result

    old_sites, new_sites = sites(raw, len(old_args)), sites(optimized, len(new_args))
    if not old_sites or old_sites.keys() != new_sites.keys():
        raise Unsupported('optimized SIMD callers changed: ' + name)
    mapping = None
    for parent, operands in new_sites.items():
        indices = []
        for operand in operands:
            matches = [i for i, previous in enumerate(old_sites[parent]) if previous == operand]
            if len(matches) != 1:
                raise Unsupported('ambiguous optimized SIMD argument mapping: ' + name)
            indices.append(matches[0])
        if len(set(indices)) != len(indices) or mapping is not None and mapping != indices:
            raise Unsupported('inconsistent optimized SIMD argument mapping: ' + name)
        mapping = indices
    restored = []
    used_names = set(re.findall(r'%[\w.$-]+', after))
    for i, arg in enumerate(old_args):
        if i in mapping:
            retained = new_args[mapping.index(i)]
            if type_and_value(arg)[0] != type_and_value(retained)[0]:
                raise Unsupported('optimizer changed SIMD argument ABI: ' + name)
            param = re.search(r'%[\w.$-]+$', retained).group()
        else:
            param = '%optimized_unused_' + str(i)
            while param in used_names:
                param += '_'
        used_names.add(param)
        restored.append(re.sub(r'%[\w.$-]+$', param, arg))
    header = new_header[:new_match.start(1)] + ', '.join(restored) + new_header[new_match.end(1):]
    return header + '\n' + after.split('\n', 1)[1]


def vector_bodies(raw, optimized):
    pattern = r'^define [^\n]*@([^ (]+)[^\n]*\{\n.*?^}'
    old = {m[1]: m[0] for m in re.finditer(pattern, raw, re.M | re.S) if 'ptc_simdvf' in m[0].splitlines()[0]}
    new = {m[1]: m[0] for m in re.finditer(pattern, optimized, re.M | re.S) if 'ptc_simdvf' in m[0].splitlines()[0]}
    if not old or not set(old).issubset(new):
        raise Unsupported('optimized output is missing an original SIMD function')

    def abi(body):
        header = body.splitlines()[0]
        match = re.search(r'ptc_simdvf (void) @[^ (]+\((.*)\)', header)
        if not match:
            raise Unsupported('unsupported optimized SIMD return ABI')
        return [type_and_value(arg)[0] for arg in split_top(match[2]) if arg]

    for name, body in old.items():
        if abi(body) != abi(new[name]):
            new[name] = restore_removed_arguments(name, body, new[name], raw, optimized)
    return re.sub(pattern, lambda m: new[m[1]] if m[1] in old else m[0], raw, flags=re.M | re.S)


def verified_vectors(raw, files):
    required = ('optimized_llvm', 'optimized_proof', 'optimized_binary', 'optimizer', 'optimizer_meta',
                'optimizer_libdevice')
    if not all(k in files for k in required):
        raise Unsupported('missing optimized SIMD evidence')
    for role in (*required, 'llvm', 'binary'):
        if sha256(files[role]['path']) != files[role]['sha256']:
            raise Unsupported('optimized SIMD evidence changed: ' + role)
    proof = json.loads(Path(files['optimized_proof']['path']).read_text())
    if proof.get('version') != VERSION or proof.get('device_executed') is not False:
        raise Unsupported('invalid optimized SIMD proof')
    for role in ('llvm', 'binary', 'optimized_llvm', 'optimized_binary', 'optimizer', 'optimizer_meta',
                 'optimizer_libdevice'):
        if proof.get('files', {}).get(role) != files[role]:
            raise Unsupported('optimized SIMD proof identity differs: ' + role)
    original = executable_sections(files['binary']['path'])
    rebuilt = executable_sections(files['optimized_binary']['path'])
    if original != rebuilt or original != proof.get('executable_sections'):
        raise Unsupported('optimizer reproduction does not match executable sections')
    if hashlib.sha256(raw.encode()).hexdigest() != files['llvm']['sha256']:
        raise Unsupported('analysis LLVM differs from optimizer input')
    text = vector_bodies(raw, Path(files['optimized_llvm']['path']).read_text())
    return text, dict(version=VERSION, proof=files['optimized_proof'], binary=files['binary'],
                      executable_sections_matched=True,
                      limitations='optimized LLVM plus assumed latency/control/queue costs; not final ISA scheduling')


def prepare(candidate, output, compiler, llvm_link):
    """Run only outside the selector audit, in a new preparation directory."""
    import os
    import subprocess
    output.mkdir(parents=True, exist_ok=False)
    files = candidate['files']
    for role in ('llvm', 'binary', 'compiler', 'final_ir'):
        if sha256(files[role]['path']) != files[role]['sha256']:
            raise ValueError('candidate input changed: ' + role)
    target = json.loads(Path(files['metadata']['path']).read_text())['target']['arch']
    if target != 'Ascend950PR_9579':
        raise ValueError('optimized compiler profile requires Ascend950PR_9579')
    final_ir = Path(files['final_ir']['path']).read_text()
    from vv_mixed_model import meta_path
    meta = meta_path(final_ir)
    libdevice = Path(files['compiler']['path']).parent.parent.parent / 'lib/libdevice.10.bc'
    inputs = dict(llvm=files['llvm'], binary=files['binary'], optimizer=record(compiler), optimizer_meta=record(meta),
                  optimizer_libdevice=record(libdevice))
    # A5 HIVMCMainA5.cpp's CPU device-compilation profile. Equality of the
    # rebuilt executable sections is required; a different backend fails closed.
    args = [
        str(compiler), '-c', '-x', 'ir', files['llvm']['path'], '-o',
        str(output / 'kernel.o'), '--cce-aicore-arch=dav-c310-vec', '--cce-aicore-only', '-O2',
        '-cce-bitcode-is-aicore', '-Wno-override-module', '-mllvm', '--cce-vf-auto-sync=global', '-mllvm',
        '-cce-aicore-dcci-insert-for-scalar=false', '-mllvm', '-enable-module-inliner', '-mllvm',
        '-replexitval=unusedindvarinloop', '-mllvm', '-cce-vf-aa-between-iters=true', '-mllvm',
        '-cce-vf-enable-loop-fusion=false', '--cce-disable-asc-reserved-ubuf', '-mllvm', '-cce-vf-stack-size=0x2000',
        '--cce-long-scbz=true', '-cce-link-aicore-ll-module',
        str(meta), '-cce-link-aicore-ll-module',
        str(libdevice)
    ]
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES='', ASCEND_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
    commands = []

    def run(command, name):
        commands.append(command)
        with (output / (name + '.stdout')).open('w') as stdout, (output / (name + '.stderr')).open('w') as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, env=env, check=True, timeout=240)

    run(args, 'object')
    sections = executable_sections(files['binary']['path'])
    if sections != executable_sections(output / 'kernel.o'):
        raise ValueError('CPU reproduction differs from the candidate executable sections')
    emit = args[:]
    emit.insert(2, '-emit-llvm')
    emit[emit.index('-o') + 1] = str(output / 'kernel.bc')
    run(emit, 'optimized')
    run([str(llvm_link), '-S', str(output / 'kernel.bc'), '-o', str(output / 'optimized.ll')], 'decode')
    for item in inputs.values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('input changed during optimization')
    vector_bodies(Path(files['llvm']['path']).read_text(), (output / 'optimized.ll').read_text())
    inputs.update(optimized_binary=record(output / 'kernel.o'), optimized_llvm=record(output / 'optimized.ll'))
    proof = dict(version=VERSION, files=inputs, executable_sections=sections, commands=commands,
                 llvm_link=record(llvm_link), device_executed=False, timing_labels_read=False)
    (output / 'proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    files.update(inputs, optimized_proof=record(output / 'proof.json'))


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--llvm-link', type=Path, required=True)
    p.add_argument('--scope', choices=('mixed', 'all'), default='mixed',
                   help='Prepare mixed families only, or every VV candidate family')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(a.manifest.read_text())
    from vv_candidate_plan import plan_candidates
    for case in manifest['cases']:
        planned = plan_candidates(case['contract']['grid'], case['candidates'])['keys']
        family = [case['candidates'][k] for k in planned if case['candidates'][k].get('resource_checked')]
        mixed = any(
            json.loads(Path(c['files']['metadata']['path']).read_text()).get('parallel_mode') == 'mix_simd_simt'
            for c in family)
        if a.scope == 'mixed' and not mixed:
            continue
        # Prepare the baseline as well: never compare old and new cost models.
        for key in planned:
            candidate = case['candidates'][key]
            if candidate.get('resource_checked'):
                prepare(candidate, a.output / case['id'] / key, a.compiler, a.llvm_link)
                print(case['id'], key, 'EXECUTABLE_SECTIONS_MATCHED', flush=True)
                (a.output / 'manifest.partial.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
