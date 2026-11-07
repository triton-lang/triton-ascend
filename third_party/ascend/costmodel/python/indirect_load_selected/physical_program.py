"""Frozen expression evaluation; no IO or device access."""
import math


def evaluate(program, features):
    """Independent expression evaluator returns exact effective <=8-term pieces."""
    nodes = program['nodes']

    def run(i):
        while nodes[i]['kind'] in ('le', 'eq'):
            n = nodes[i]
            v = features[n['feature']]
            yes = v <= n['value'] if n['kind'] == 'le' else int(v) == n['value']
            i = n['yes'] if yes else n['no']
        n = nodes[i]
        kind = n['kind']
        if kind in ('linear', 'exp'): return dict(terms=dict(n['terms']), floor=1., link=kind)
        if kind == 'scale':
            p = run(n['child'])
            assert p['link'] == 'linear'
            factor = n['value']
            p['terms'] = {k: v * factor
                          for k, v in p['terms'].items()}
            p['floor'] = max(1., p['floor'] * factor)
            return p
        if kind == 'blend':
            p = run(n['first'])
            q = run(n['second'])
            assert p['link'] == q['link'] == 'linear' and set(q['terms']) == {'one'} and 'one' in p['terms']
            constant = max(q['floor'], q['terms']['one'])
            a = n['value']
            p['terms'] = {k: v * a
                          for k, v in p['terms'].items()}
            p['terms']['one'] += (1 - a) * constant
            p['floor'] = a * p['floor'] + (1 - a) * constant
            assert len(p['terms']) <= 8
            return p
        raise ValueError('Invalid expression instruction')

    p = run(program['root'])
    v = math.fsum(features[k] * c for k, c in p['terms'].items())
    return max(p['floor'],
               math.exp(max(-20., min(30., v))) if p['link'] == 'exp' else v), len(p['terms'])
