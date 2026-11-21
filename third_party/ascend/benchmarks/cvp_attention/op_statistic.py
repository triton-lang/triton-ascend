"""Read one exact hardware-operation aggregate without inventing task samples."""
import csv
import hashlib
import io
import math
from pathlib import Path
import re

REPRESENTATION = 'op-statistic-aggregate'
TIME_COLUMNS = ('Total Time(us)', 'Min Time(us)', 'Avg Time(us)', 'Max Time(us)')
REQUIRED_COLUMNS = ('Device_id', 'OP Type', 'Core Type', 'Count', *TIME_COLUMNS)


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(label + ' must be a finite positive number of microseconds')
    return value


def validate_op_statistic_block(block, expected_count=10):
    """Check reported aggregate consistency; callers must audit the actual CSV."""
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError('Expected a positive operation count')
    if not isinstance(block, dict) or block.get('representation') != REPRESENTATION:
        raise ValueError('Expected an explicit op_statistic aggregate')
    count = block.get('task_count')
    if type(count) is not int or count != expected_count:
        raise ValueError('Unexpected attention operation count')
    total = _positive(block.get('total_duration_us'), 'Total')
    minimum = _positive(block.get('min_duration_us'), 'Minimum')
    maximum = _positive(block.get('max_duration_us'), 'Maximum')
    reported = _positive(block.get('reported_avg_duration_us'), 'Reported average')
    average = _positive(block.get('mean_duration_us'), 'Mean')
    mean = total / count
    # CANN exports rounded decimal microseconds. This allows its last decimal,
    # without allowing a missing task or a materially inconsistent average.
    tolerance = max(0.001, abs(mean) * 1e-12)
    if abs(average - mean) > tolerance or abs(reported - mean) > tolerance:
        raise ValueError('Total/count and reported average disagree')
    if minimum > maximum or mean < minimum - tolerance or mean > maximum + tolerance:
        raise ValueError('Mean is outside the reported minimum/maximum')
    if type(block.get('device_id')) is not int or block['device_id'] < 0:
        raise ValueError('Record the selected physical device ID')
    if not all(isinstance(block.get(key), str) and block[key] for key in ('op_type', 'core_type', 'source_path')):
        raise ValueError('Record the exact operator, core type and CSV path')
    if not isinstance(block.get('source_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', block['source_sha256']):
        raise ValueError('Record the complete source CSV SHA256')
    if type(block.get('source_record_index')) is not int or block['source_record_index'] < 0:
        raise ValueError('Record the zero-based CSV data-record index')
    return mean


def parse_op_statistic_bytes(data, *, source_path, device_id, op_type, core_type, expected_count=10):
    """Select exactly one row by device, full operator name and core type."""
    if type(device_id) is not int or device_id < 0 or type(expected_count) is not int or expected_count <= 0:
        raise ValueError('Expected explicit device and positive operation count')
    if not isinstance(data, bytes) or not op_type or not core_type or not str(source_path):
        raise ValueError('Supply CSV bytes and an exact operator/core/path selection')
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    headers = reader.fieldnames
    if headers is None or len(headers) != len(set(headers)) or not set(REQUIRED_COLUMNS) <= set(headers):
        raise ValueError('Missing, duplicate or unsupported CSV headers; microseconds are required')
    selected = []
    for index, row in enumerate(reader):
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Malformed CSV record')
        if (row['Device_id'].strip() == str(device_id) and row['OP Type'].strip() == op_type
                and row['Core Type'].strip() == core_type):
            selected.append((index, row))
    if len(selected) != 1:
        raise ValueError('Expected exactly one matching attention row, got ' + str(len(selected)))
    index, row = selected[0]
    if not re.fullmatch(r'[0-9]+', row['Count'].strip()):
        raise ValueError('Attention count must be an integer')
    try:
        total, minimum, average, maximum = [float(row[name]) for name in TIME_COLUMNS]
    except ValueError as error:
        raise ValueError('Invalid microsecond duration') from error
    count = int(row['Count'])
    if count != expected_count:
        raise ValueError('Unexpected attention operation count')
    block = dict(representation=REPRESENTATION, device_id=device_id, op_type=op_type,
                 core_type=core_type, task_count=count, total_duration_us=total,
                 mean_duration_us=total / count, reported_avg_duration_us=average,
                 min_duration_us=minimum, max_duration_us=maximum,
                 source_path=str(source_path), source_sha256=hashlib.sha256(data).hexdigest(),
                 source_record_index=index, source_row=dict(row))
    validate_op_statistic_block(block, expected_count)
    return block


def read_op_statistic(path, *, device_id, op_type, core_type, expected_count=10):
    path = Path(path).resolve(strict=True)
    return parse_op_statistic_bytes(path.read_bytes(), source_path=str(path), device_id=device_id,
                                    op_type=op_type, core_type=core_type, expected_count=expected_count)
