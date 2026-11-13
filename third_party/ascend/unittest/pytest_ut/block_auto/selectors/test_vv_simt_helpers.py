import unittest
from pathlib import Path

from backend_cost_model import ASSUMPTIONS, Unsupported, parse_module
from vv_contention_model import ContentionProfile, schedule
from vv_mixed_model import MixedSchedule
from vv_simt_helpers import VerifiedSimtHelpers, decode_dimensions, validate_abi

DATA = Path(__file__).with_name('simt_test_data')
PARAMETERS = dict(simt_transaction_bytes=64, simt_transition_cycles=32, simt_instruction_cycles=1)
GATHER = 'indirect_load_nonvolatile_2d_float_int64_t'
SCAN = 'cumsum_1d_float_dim0'


def gather_args():
    return [None, None, 0, 0, 0] + [None, None, 0, 8, 8, 8, 1] * 4


def scan_args():
    return [None, None, 0, 512, 1] * 2 + [None, None, 0, 32, 1] + [0]


class SimtHelperTests(unittest.TestCase):

    def model(self, fixture, text=None):
        return VerifiedSimtHelpers(text or (DATA / (fixture + '.ll')).read_text(), {}, PARAMETERS)

    def test_named_entry_is_actual_entry(self):
        f = parse_module('define void @f() {\nentry:\n  ret void\n}')["f"]
        self.assertEqual(f.entry, 'entry')
        self.assertEqual(list(f.blocks), ['entry'])

    def test_launch_dimensions_and_invalid_bounds(self):
        self.assertEqual(decode_dimensions(32 | (16 << 16) | (1 << 32)), (32, 16, 1))
        for info in [0, -1, 1 << 48, 4097 | (1 << 16) | (1 << 32)]:
            with self.subTest(info=info), self.assertRaises(Unsupported):
                decode_dimensions(info)

    def test_gather_counts_index_and_payload_widths(self):
        r = self.model('gather2d').profile(GATHER, gather_args())
        self.assertEqual(r['work_items'], 64)
        self.assertEqual(r['counts']['gm_loads'], 64)
        self.assertEqual(r['counts']['gm_load_bytes'], 64 * 4)
        self.assertEqual(r['transaction_bytes_upper'], 64 * 64)
        self.assertEqual(r['warps'], 2)
        self.assertEqual(r['counts']['ub_store_bytes'], 64 * 4)
        self.assertEqual(r['counts']['ub_load_bytes'], 64 * (1 + 8))
        self.assertGreater(r['cycles'], 0)

    def test_scan_has_shuffle_barrier_and_no_gm_traffic(self):
        r = self.model('scan512').profile(SCAN, scan_args())
        self.assertEqual(r['dimensions'], (32, 16, 1))
        self.assertEqual(r['counts']['workitem_barriers'], 512)
        self.assertEqual(r['warps'], 16)
        self.assertEqual(r['warp_counts']['workitem_barriers'], 16)
        self.assertLess(r['warp_counts']['instruction_steps'], r['counts']['instruction_steps'])
        self.assertGreater(r['counts']['shuffles'], 512)
        self.assertEqual(r['transaction_bytes_upper'], 0)
        self.assertGreaterEqual(r['counts']['ub_store_bytes'], 512 * 4)

    def test_unknown_helper_is_not_zero_cost(self):
        self.assertIsNone(self.model('scan512').profile('unknown_helper', []))

    def test_linked_gather_joins_pipes_and_shares_gm_bandwidth(self):
        s = MixedSchedule(ASSUMPTIONS, None, simt_helpers=self.model('gather2d'))
        s.vf(dict(cycles=100, counts={}))
        vector = s.pipe_refs[1][0]
        s.dma('load', 1024)
        dma = s.pipe_refs[4][0]
        self.assertTrue(s.extra_call(GATHER, gather_args()))
        simt = len(s.events) - 1
        self.assertTrue({vector, dma}.issubset(dict(s.events[simt]['dependencies'])))
        s.vf(dict(cycles=10, counts={}))
        self.assertIn(simt, dict(s.events[-1]['dependencies']))
        result = schedule([s.result()], 1, ContentionProfile())
        self.assertEqual(result['simt_gm_transaction_bytes_upper'], 64 * 64)
        self.assertEqual(result['dma_requests'], 1)

    def test_linked_scan_does_not_wait_for_unrelated_gm_bus(self):
        scan = MixedSchedule(ASSUMPTIONS, None, simt_helpers=self.model('scan512'))
        scan.extra_call(SCAN, scan_args())
        other = MixedSchedule(ASSUMPTIONS, None)
        other.dma('load', 10**8)
        scan_trace = scan.result()
        isolated = schedule([scan_trace], 1, ContentionProfile())
        concurrent = schedule([other.result(), scan_trace], 2, ContentionProfile())
        self.assertEqual(concurrent['block_completion'][1] - concurrent['block_release'][1],
                         isolated['block_completion'][0] - isolated['block_release'][0])
        self.assertEqual(concurrent.get('simt_gm_transaction_bytes_upper', 0), 0)

    def test_invalid_descriptors_fail_before_cached_profile(self):
        m = self.model('gather2d')
        m.profile(GATHER, gather_args())
        for index, value in [(8, None), (9, 0), (11, 2), (15, 9), (7, -1)]:
            args = gather_args()
            args[index] = value
            with self.subTest(index=index), self.assertRaises(Unsupported):
                m.profile(GATHER, args)
        args = scan_args()
        args[8] = 513
        with self.assertRaises(Unsupported):
            validate_abi(SCAN, args)
        args = scan_args()
        args[15] = None
        with self.assertRaises(Unsupported):
            validate_abi(SCAN, args)

    def test_missing_linked_function_rejected(self):
        with self.assertRaisesRegex(Unsupported, 'definition missing'):
            self.model('gather2d', '; empty module').profile(GATHER, gather_args())

    def test_dynamic_launch_configuration_rejected(self):
        text = (DATA / 'scan512.ll').read_text().replace('i64 4296015904', 'i64 %unknown')
        with self.assertRaisesRegex(Unsupported, 'statically configured'):
            self.model('scan512', text).profile(SCAN, scan_args())

    def test_unknown_simt_intrinsic_rejected(self):
        text = (DATA / 'scan512.ll').read_text().replace('llvm.hivm.shfl.idx.i32', 'llvm.hivm.unknown.i32')
        with self.assertRaisesRegex(Unsupported, 'unmodelled call'):
            self.model('scan512', text).profile(SCAN, scan_args())

    def test_unmodelled_data_dependent_control_rejected(self):
        text = (DATA / 'gather2d.ll').read_text().replace('load i8,', 'load i16,').replace('icmp eq i8', 'icmp eq i16')
        with self.assertRaisesRegex(Unsupported, 'data-dependent branch'):
            self.model('gather2d', text).profile(GATHER, gather_args())


if __name__ == '__main__':
    unittest.main()
