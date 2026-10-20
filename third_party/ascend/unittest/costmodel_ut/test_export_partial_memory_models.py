"""Guard the exported affine model boundary and complexity limits."""
import importlib.util
import math
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'costmodel/utils/export_partial_memory_models.py'
SPEC = importlib.util.spec_from_file_location('partial_export', SCRIPT)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def partition():
    return dict(terms=['one', 'E'], splits=['bits', 'unused'],
                groups={'1': [2., 3.], '2': [5., 7.]},
                nodes=[dict(feature=0, threshold=16., left=1, right=2),
                       dict(feature=-1), dict(feature=-1)])


class ShortFormulaExportTest(unittest.TestCase):
    def test_leaf_formulas_and_selector_equality(self):
        formulas, nodes = EXPORT.normalize(partition())
        self.assertEqual(len(formulas), 2)
        name, threshold, left, right = nodes[0]
        self.assertEqual(name, 'bits')
        for value, expected in [(math.nextafter(16., -math.inf), 0),
                                (16., 0), (math.nextafter(16., math.inf), 1)]:
            ref = left if value <= threshold else right
            self.assertEqual(-1-ref, expected)

    def test_formula_complexity_is_bounded(self):
        model = partition()
        model['terms'] = ['E'] * 11
        model['groups'] = {'1': [1.] * 11, '2': [1.] * 11}
        with self.assertRaises(ValueError):
            EXPORT.normalize(model)
        with self.assertRaises(ValueError):
            EXPORT.normalize(dict(groups={str(i): [1.] for i in range(129)}))
        with self.assertRaises(ValueError):
            EXPORT.normalize(dict(trees=[], groups={}))

    def test_nonfinite_and_cyclic_models_are_rejected(self):
        model = partition()
        model['groups']['1'][0] = float('nan')
        with self.assertRaises(ValueError):
            EXPORT.normalize(model)
        model = partition()
        model['nodes'][0]['left'] = 0
        with self.assertRaises(ValueError):
            EXPORT.normalize(model)

    def test_explicit_groups_require_fallback(self):
        model = dict(family=['bits', 'regular_rank', 'tail_kind', 'tile_large'],
                     groups={'all': dict(terms=['one'], coefficients=[2.])})
        self.assertEqual(EXPORT.normalize(model)[1], [])
        model['groups'] = {'8_0_0_0': dict(terms=['one'], coefficients=[2.])}
        with self.assertRaises(ValueError):
            EXPORT.normalize(model)


if __name__ == '__main__':
    unittest.main()
