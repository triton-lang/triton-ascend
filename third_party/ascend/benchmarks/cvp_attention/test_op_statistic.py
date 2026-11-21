"""Synthetic CSV fixtures verify selection/statistics, never NPU performance."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from op_statistic import parse_op_statistic_bytes, read_op_statistic, validate_op_statistic_block
from paired_stats import summarize_paired_rounds
from test_paired_stats import cohort

HEADER = 'Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)\n'


def fixture(device=0, op='_attn_fwd', core='AI_CORE', count=10, total=1120, minimum=80, average=112, maximum=400):
    return (HEADER + f'{device},{op},{core},{count},{total},{minimum},{average},{maximum},100\n').encode()


def parse(data=None, path='synthetic.csv', **kwargs):
    options = dict(device_id=0, op_type='_attn_fwd', core_type='AI_CORE')
    options.update(kwargs)
    return parse_op_statistic_bytes(fixture() if data is None else data, source_path=path, **options)


def aggregate_cohort():
    data = cohort(cvp=112, ascend=100)
    for row in data['rounds']:
        for arm, op, core, total, minimum, average, maximum in (
            ('cvpipeline', '_attn_fwd', 'AI_CORE', 1120, 80, 112, 400),
            ('ascendc', 'FlashAttentionScore', 'MIX_AIC', 1000, 100, 100, 100),
        ):
            row.pop(arm + '_durations_us')
            row[arm + '_op_statistic'] = parse(fixture(op=op, core=core, total=total,
                                                       minimum=minimum, average=average, maximum=maximum),
                                                   path=f'synthetic-round-{row["round_index"]}-{arm}.csv',
                                                   op_type=op, core_type=core)
    return data


class OpStatisticTest(unittest.TestCase):
    def test_total_divided_by_real_count_without_inventing_samples(self):
        block = parse()
        self.assertEqual(validate_op_statistic_block(block), 112)
        self.assertEqual(block['task_count'], 10)
        self.assertNotIn('durations_us', block)

    def test_exact_operator_device_core_selection_ignores_prefix_collisions(self):
        data = fixture() + b'0,attn_fwd,AI_CORE,10,9000,900,900,900,100\n1,_attn_fwd,AI_CORE,10,8000,800,800,800,100\n0,_attn_fwd,MIX_AIC,10,7000,700,700,700,100\n'
        self.assertEqual(parse(data)['mean_duration_us'], 112)

    def test_missing_and_duplicate_matching_rows_reject(self):
        for data in (fixture(op='attn_fwd'), fixture() + fixture().split(b'\n', 1)[1]):
            with self.assertRaises(ValueError): parse(data)

    def test_missing_wrong_unit_and_duplicate_headers_reject(self):
        for data in (fixture().replace(b'Avg Time(us)', b'Avg Time(ms)'),
                     fixture().replace(b'Ratio(%)', b'Count')):
            with self.assertRaises(ValueError): parse(data)

    def test_unexpected_task_counts_and_noninteger_count_reject(self):
        for count in (0, 9, 11, '10.0', 'True'):
            with self.assertRaises(ValueError): parse(fixture(count=count))

    def test_nonfinite_negative_and_zero_durations_reject(self):
        for total in ('nan', 'inf', 0, -1):
            with self.assertRaises(ValueError): parse(fixture(total=total))

    def test_total_average_and_extrema_must_agree(self):
        for data in (fixture(total=1100), fixture(minimum=120), fixture(maximum=100)):
            with self.assertRaises(ValueError): parse(data)

    def test_cann_decimal_rounding_is_accepted(self):
        block = parse(fixture(total=100.004, average=10.0, minimum=9, maximum=11))
        self.assertAlmostEqual(block['mean_duration_us'], 10.0004)

    def test_utf8_bom_file_hash_and_resolved_path_preserved(self):
        raw = b'\xef\xbb\xbf' + fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'op_statistic.csv'
            path.write_bytes(raw)
            block = read_op_statistic(path, device_id=0, op_type='_attn_fwd', core_type='AI_CORE')
            self.assertEqual(block['source_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(block['source_path'], str(path.resolve()))
            self.assertEqual(block['source_record_index'], 0)

    def test_malformed_csv_record_rejects(self):
        with self.assertRaises(ValueError): parse(fixture() + b'0,_attn_fwd\n')

    def test_aggregate_provenance_and_count_are_required(self):
        for field, value in (('source_path', ''), ('source_sha256', 'missing'),
                             ('source_record_index', -1), ('task_count', True)):
            block = parse(); block[field] = value
            with self.assertRaises(ValueError): validate_op_statistic_block(block)

    def test_real_aggregate_means_feed_paired_statistics(self):
        data = aggregate_cohort()
        result = summarize_paired_rounds(data['case'], data['rounds'])
        self.assertEqual(result['sample_representation'], 'op-statistic-aggregate')
        self.assertEqual(result['median_us']['cvpipeline'], 112)
        self.assertFalse(result['strict_target_met'])

    def test_same_csv_record_cannot_be_reused_for_another_round(self):
        data = aggregate_cohort()
        data['rounds'][1]['cvpipeline_op_statistic'] = copy.deepcopy(data['rounds'][0]['cvpipeline_op_statistic'])
        with self.assertRaises(ValueError): summarize_paired_rounds(data['case'], data['rounds'])

    def test_different_device_or_operator_rejects(self):
        for field, value in (('device_id', 1), ('op_type', 'Add'), ('core_type', 'CHANGED')):
            data = aggregate_cohort()
            data['rounds'][2]['cvpipeline_op_statistic'][field] = value
            with self.assertRaises(ValueError): summarize_paired_rounds(data['case'], data['rounds'])

    def test_both_representations_missing_or_mixed_reject(self):
        for mode in ('both', 'missing', 'mixed'):
            data = aggregate_cohort()
            row = data['rounds'][0]
            if mode == 'both': row['cvpipeline_durations_us'] = [112] * 10
            if mode == 'missing': row.pop('cvpipeline_op_statistic')
            if mode == 'mixed':
                row.pop('cvpipeline_op_statistic'); row['cvpipeline_durations_us'] = [112] * 10
            with self.assertRaises(ValueError): summarize_paired_rounds(data['case'], data['rounds'])


if __name__ == '__main__':
    unittest.main()
