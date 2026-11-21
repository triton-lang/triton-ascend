"""The hardware gate must reject incomplete or ambiguous compiler evidence."""
import unittest

from stage_evidence import pipeline_snapshots


def capture():
    module = ('module {\n'
              '  func.func @stage_chain() {\n'
              '    return\n'
              '  }\n'
              '}\n')
    phases = [('After', 'CVPipelinePrepare', 'hivm-cvp-plan-prepare'),
              ('Before', 'VerifyCVPipeline', 'hivm-verify-cvp-plan'),
              ('After', 'VerifyCVPipeline', 'hivm-verify-cvp-plan')]
    return ''.join(f'// -----// IR Dump {when} {name} ({flag}) '
                   "('func.func' operation: @stage_chain) //----- //\n" + module for when, name, flag in phases)


class StageEvidenceTest(unittest.TestCase):

    def test_complete_snapshots_exclude_trailing_compiler_messages(self):
        snapshots = pipeline_snapshots(capture() + 'compiler: completed\n')
        self.assertEqual(set(snapshots), {'prepared', 'scheduled', 'verified'})
        self.assertTrue(all(text.endswith('}\n') for text in snapshots.values()))
        self.assertTrue(all('compiler: completed' not in text for text in snapshots.values()))

    def test_missing_schedule_proof_fails(self):
        text = capture().replace('IR Dump Before VerifyCVPipeline', 'IR Dump After UnrelatedPass')
        with self.assertRaises(AssertionError):
            pipeline_snapshots(text)

    def test_repeated_attempts_are_ambiguous(self):
        with self.assertRaises(AssertionError):
            pipeline_snapshots(capture() * 2)

    def test_truncated_verified_module_fails(self):
        with self.assertRaises(AssertionError):
            pipeline_snapshots(capture().rstrip()[:-1])

    def test_wrong_kernel_cannot_supply_proof(self):
        with self.assertRaises(AssertionError):
            pipeline_snapshots(capture().replace('@stage_chain(', '@different_kernel('))


if __name__ == '__main__':
    unittest.main()
