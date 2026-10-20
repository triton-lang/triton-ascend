#!/usr/bin/env python3
"""Export one short affine formula per region; datasets remain local.
Usage: export_partial_memory_models.py OUTPUT SIMT_LOAD SIMT_STORE SIMD_LOAD SIMD_STORE
Each input contains `model`. Ensembles are not accepted.
"""
import hashlib
import json
import math
import pathlib
import sys

MAX_TERMS = 10
MAX_FORMULAS = 128


def normalize(model):
    """Turn the leaves of one selector into short formula references."""
    if 'trees' in model or 'models' in model:
        raise ValueError('Ensembles are not short affine models')
    groups = model['groups']
    if not 1 <= len(groups) <= MAX_FORMULAS:
        raise ValueError('Formula count exceeds the explicit complexity limit')
    formulas, refs = [], {}
    for key, group in sorted(groups.items()):
        terms = group['terms'] if isinstance(group, dict) else model['terms']
        coefficients = group['coefficients'] if isinstance(group, dict) else group
        if not 1 <= len(terms) <= MAX_TERMS or len(terms) != len(coefficients):
            raise ValueError('Invalid short formula length')
        if not all(math.isfinite(c) for c in coefficients):
            raise ValueError('Nonfinite coefficient')
        refs[key] = -1 - len(formulas)
        formulas.append((key, terms, coefficients))
    if not model.get('nodes'):
        if model.get('family') != ['bits', 'regular_rank', 'tail_kind', 'tile_large'] or 'all' not in refs:
            raise ValueError('Unsupported group selectors or missing fallback')
        return formulas, []
    nodes, active = [], set()

    def visit(index):
        if index in active or not 0 <= index < len(model['nodes']):
            raise ValueError('Invalid selector graph')
        node = model['nodes'][index]
        if node['feature'] < 0:
            return refs[str(index)]
        active.add(index)
        n = len(nodes)
        nodes.append(None)
        name, threshold = model['splits'][node['feature']], node['threshold']
        if not math.isfinite(threshold):
            raise ValueError('Nonfinite selector threshold')
        left, right = visit(node['left']), visit(node['right'])
        nodes[n] = (name, threshold, left, right)
        active.remove(index)
        return n

    root = visit(0)
    if root != 0 or len(nodes) != len(formulas) - 1:
        raise ValueError('Expected one binary partition with a split root')
    return formulas, nodes


def export(paths):
    if len(paths) != 4:
        raise ValueError('Exactly four frozen models are required')
    models, names = [], set()
    for path in paths:
        raw = path.read_bytes()
        formulas, nodes = normalize(json.loads(raw)['model'])
        models.append((hashlib.sha256(raw).hexdigest(), formulas, nodes))
        names.update(n for _, terms, _ in formulas for n in terms)
        names.update(n for n, _, _, _ in nodes)
    names = sorted(names)
    feature = {name: i for i, name in enumerate(names)}
    text = ['// Generated short affine formulas; no residual ensembles.',
            '// Model order: SIMT load/store, SIMD load/store. Units: SYS_CNT ticks.',
            '// Selectors use <=. Negative references encode -1-formula index.',
            'static const char *const kFeatureNames[] = {']
    text += ['  ' + json.dumps(name) + ',' for name in names]
    text.append('};')
    for kind, (sha, formulas, nodes) in enumerate(models):
        text.append('// Frozen model {} SHA256: {}'.format(kind, sha))
        text.append('static const Formula kFormulas{}[] = {{'.format(kind))
        for key, terms, coefficients in formulas:
            text.append('  {{{}, {{{}}}, {{{}}}, {}}},'.format(json.dumps(key),
                ', '.join(str(feature[n]) for n in terms),
                ', '.join(repr(c) for c in coefficients), len(terms)))
        text.append('};')
        if nodes:
            text.append('static const Decision kDecisions{}[] = {{'.format(kind))
            text += ['  {{{}, {}, {}, {}}},'.format(feature[n], repr(c), left, right)
                     for n, c, left, right in nodes]
            text.append('};')
    text.append('static const ShortModel kModels[] = {')
    text += ['  {{{}, kFormulas{}, {}}},'.format('kDecisions' + str(i) if nodes else 'nullptr', i, len(formulas))
             for i, (_, formulas, nodes) in enumerate(models)]
    text.extend(['};', ''])
    return '\n'.join(text)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    pathlib.Path(sys.argv[1]).write_text(export([pathlib.Path(p) for p in sys.argv[2:]]))
