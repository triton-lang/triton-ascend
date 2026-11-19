"""Collect the original attention profiler recipe and six paired hardware blocks.

Importing this module does not import torch or initialize an NPU. The CLI needs
a sealed experiment configuration, a fresh admission file and an independently
qualified original-kernel case. Raw profiler output is always retained.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import time
import traceback

from op_statistic import read_op_statistic
from paired_stats import ARMS, PRODUCTION_SHAPES, summarize_paired_rounds

OPERATORS = {'cvpipeline': '_attn_fwd', 'ascendc': 'FlashAttentionScore'}
SCHEDULE = dict(wait=1, warmup=1, active=10, repeat=1, skip_first=1)


def profile_block(torch, torch_npu, call, output, guard, *, initialized=None,
                  sleeper=time.sleep, progress=None):
    """One initialization and exactly13calls, matching run_profiler_swap.py."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    guard()
    initial = call()
    torch.npu.synchronize()
    if initialized:
        initialized(initial)
    initial = None
    experimental = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
    latest = None
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False, record_shapes=False, profile_memory=False,
            schedule=torch_npu.profiler.schedule(**SCHEDULE),
            experimental_config=experimental,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output))) as prof:
        for index in range(13):
            guard()
            latest = call()
            torch.npu.synchronize()
            sleeper(1)
            prof.step()
            if progress:
                progress(index + 1)
    return latest, dict(initialization_calls=1, profile_calls=13,
                        selected_active_calls=10, schedule=dict(SCHEDULE),
                        sleep_seconds=1, synchronize_after_each_call=True,
                        timing_source='hardware profiler CSV; host elapsed is not a kernel timing')


def wait_csv(output, *, timeout=90, clock=time.monotonic, sleeper=time.sleep):
    """Allow profiler analysis to finish without reusing an older CSV."""
    deadline = clock() + timeout
    while True:
        files = list(Path(output).rglob('op_statistic.csv'))
        if len(files) > 1:
            raise ValueError('More than one profiler CSV in this new block directory')
        if files:
            return files[0]
        if clock() >= deadline:
            raise RuntimeError('Profiler did not produce an op_statistic.csv')
        sleeper(.25)


def select_block(output, *, device_id, arm, core_type, timeout=90,
                 clock=time.monotonic, sleeper=time.sleep):
    """Require one fresh CSV and one exact operator/device/core row."""
    if arm not in OPERATORS:
        raise ValueError('Unknown comparison arm')
    path=wait_csv(output,timeout=timeout,clock=clock,sleeper=sleeper)
    return read_op_statistic(path,device_id=device_id,op_type=OPERATORS[arm],
                             core_type=core_type,expected_count=10)


def infer_csv_device(output, *, arm, core_type):
    """Mapping-only calibration; never select by largest time or average."""
    path=wait_csv(output)
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows=[r for r in csv.DictReader(stream)
              if r.get('OP Type','').strip()==OPERATORS[arm] and r.get('Core Type','').strip()==core_type]
    if len(rows)!=1 or not rows[0].get('Device_id','').strip().isdigit():
        raise ValueError('Mapping needs one exact attention operator/core row')
    device=int(rows[0]['Device_id'].strip())
    select_block(output,device_id=device,arm=arm,core_type=core_type)
    return device


def qualified_native(result, case, preset, options):
    """Reject diagnostic-only, partial, different-case or unstressed results."""
    if (result.get('finished') is not True or result.get('exit') != 0
            or result.get('measurement_mode') != 'correctness_only'
            or result.get('timed_task_count') != 0):
        raise ValueError('A terminal numerical qualification is required')
    recorded = result.get('case', {})
    for key in ('case_id', 'shape', 'block_m', 'block_n'):
        if recorded.get(key) != case.get(key):
            raise ValueError('Qualification case/geometry differs: ' + key)
    numerics = result.get('numerics', [])
    if [(x.get('seed'), x.get('kind')) for x in numerics] != [(12,'normal'),(0,'normal'),(42,'normal')]:
        raise ValueError('Production qualification must contain all three declared seeds')
    for row in numerics:
        if (row.get('passed') is not True or not row.get('inputs')
                or row.get('inputs_after') != row.get('inputs')):
            raise ValueError('Missing numerical pass or unchanged-input evidence')
        variants = [x for x in row.get('variants', []) if x.get('preset') == preset]
        if len(variants) != 1 or variants[0].get('passed') is not True:
            raise ValueError('Selected preset was not numerically qualified')
    for arm in (preset, 'ascendc'):
        stress = [x for x in result.get('stress', []) if x.get('preset') == arm]
        if sorted(x.get('iteration') for x in stress) != [32,64]:
            raise ValueError(arm+' needs complete stress checkpoints at32and64')
        if not all(all(c.get('passed') and c.get('finite') for c in x.get('checks', [])) and len(x.get('checks', []))==2 for x in stress):
            raise ValueError('Stress did not pass full finite O/M')
    if result.get('stress_inputs_after') != result.get('stress_inputs') or not result.get('stress_inputs'):
        raise ValueError('Missing stress input-integrity evidence')
    if result['stress_inputs'] != numerics[0]['inputs']:
        raise ValueError('Stress inputs differ from the declared seed12numerical inputs')
    records = [x for x in result.get('native_compiles', []) if x.get('preset') == preset]
    if len(records) != 1 or records[0].get('hardware_correctness') != 'passed-for-this-case':
        raise ValueError('Missing qualified native image')
    if records[0].get('options') != options:
        raise ValueError('Qualification compiler options differ')
    return records[0]


def qualified_recipe(result, case, presets):
    """Bind the declared preset recorded by Runtime, plus its effective kwargs.

    Runtime records the full preset. launch_options removes fixed host kwargs
    only to avoid duplicate Python arguments; it is not the record's schema.
    """
    from runtime_common import launch_options
    preset=case['initial_candidate']
    declared=copy.deepcopy(presets[preset])
    effective=launch_options(declared)
    record=qualified_native(result,case,preset,declared)
    return declared,effective,record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config','recipes','qualified-result','output','admission','stop-marker'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--qualified-sha256', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--cvp-core-type', default='AI_CORE')
    parser.add_argument('--ascendc-core-type', default='AI_CORE')
    parser.add_argument('--csv-device-id', type=int,
                        help='Actual CSV device ID from a fresh mapping calibration; may differ from physical ID')
    parser.add_argument('--calibration-only', action='store_true',
                        help='Collect one pair to establish CSV device mapping; exclude its durations from adoption')
    args = parser.parse_args()
    from runtime_common import Runtime, launch_options, save_json, sha_file, KERNEL_SHA
    from admission_gate import AdmissionGate
    cfg = json.loads(args.config.read_text())
    recipes = json.loads(args.recipes.read_text())
    expected_mode='mapping-calibration' if args.calibration_only else 'hardware-operation-duration'
    if not cfg['authorized'] or cfg['measurement_mode'] != expected_mode or not cfg['stage6_excluded']:
        raise ValueError('A sealed hardware timing configuration is required')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != str(cfg['physical_npu']):
        raise ValueError('Physical device visibility differs from the configuration')
    if os.environ.get('TRITON_ALWAYS_COMPILE') != '0':
        raise ValueError('Use a fresh private cache with recompilation disabled during profiling')
    for key in ('TRITON_OVERRIDE_DIR','TRITON_KERNEL_OVERRIDE','LLVM_IR_ENABLE_DUMP','MLIR_ENABLE_DUMP'):
        if os.environ.get(key):
            raise ValueError('Timing cannot use IR overrides or exhaustive compiler dumps')
    if sha_file(cfg['wheel_path']) != cfg['wheel_sha256']:
        raise ValueError('Selected wheel checksum differs')
    helper=shutil.which('hivmc-a5')
    if not helper or str(Path(helper).resolve()) != cfg['hivmc_a5_path'] or sha_file(helper) != cfg['hivmc_a5_sha256']:
        raise ValueError('Selected downstream HIVMC tool differs')
    if sha_file(args.recipes) != cfg['recipes_sha256']:
        raise ValueError('Recipe checksum differs from the sealed configuration')
    if sha_file(args.qualified_result) != args.qualified_sha256:
        raise ValueError('Qualification checksum differs')
    selected = next(x for x in cfg['cases'] if x['case_id']==args.case_id)
    shape = tuple(selected['shape']); bn = selected['block_n']; preset = selected['initial_candidate']
    if shape not in PRODUCTION_SHAPES:
        raise ValueError('Select one of the eleven BS128/H8 production cases')
    qualified = json.loads(args.qualified_result.read_text())
    declared_options,options,expected = qualified_recipe(qualified,selected,recipes['presets'])
    if options['enable_cvp_persistent_pipeline'] and options.get('disable_tightly_coupled_buffer_reuse') is not True:
        raise ValueError('Query timing requires the protected buffer policy')
    proof = json.loads(Path(expected['qualification_path']).read_text())
    if sha_file(expected['qualification_path']) != expected['qualification_sha256']:
        raise ValueError('Native qualification proof checksum differs')
    if proof['stress_forwards'] != 64 or not proof['full_O_M_checked']:
        raise ValueError('Native proof does not cover the required output/stress scope')
    args.output.mkdir(exist_ok=False)
    if not args.calibration_only and (args.csv_device_id is None or args.csv_device_id < 0):
        raise ValueError('CSV device ID must be nonnegative')
    state = dict(case_id=args.case_id, case=dict(case_id=args.case_id,shape=list(shape),block_m=128,block_n=bn,
        dtype='float16',layout='BNSD',causal=False,scale=.5), preset=preset, options=options, declared_options=declared_options,
        pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),job_id=cfg['job_id'],nonce=cfg['nonce'],
        started_at=datetime.now(timezone.utc).isoformat(),finished=False,exit=None,
        measurement_mode=expected_mode,timing_unit='us',rounds=[],blocks=[],calibration_only=args.calibration_only,
        config_sha256=sha_file(args.config),recipes_sha256=sha_file(args.recipes),
        qualification_sha256=args.qualified_sha256,collector_sha256=sha_file(__file__),
        untimed_reference_preparation_calls=1,physical_npu=cfg['physical_npu'],csv_device_id=args.csv_device_id)
    save = lambda: save_json(args.output/'RESULT.json',state)
    save()
    guard = AdmissionGate(args.admission,args.stop_marker,cfg['physical_npu'])
    try:
        guard()
        runtime = Runtime(cfg['kernel_path'],args.output/'compiler',cfg,recipes,guard)
        torch = runtime.torch
        q,k,v = runtime.inputs(shape,12)
        inputs = [runtime.fingerprint(x) for x in (q,k,v)]
        raw = runtime.ascendc(q,k,v);torch.npu.synchronize()
        golden = raw[0],runtime.lse_from_ascendc(raw,shape);raw=None
        state['source_identity'] = runtime.source_identity
        state['input_fingerprint'] = inputs
        identities = dict(cvpipeline=dict(native_sha256=expected['native_sha256'],options=options,kernel_sha256=KERNEL_SHA,
            wheel_sha256=cfg['wheel_sha256'],frontend_sha256=runtime.tools['frontend']['sha256'],
            reader_sha256=runtime.tools.get('reader', {}).get('sha256'),
            core_sha256=runtime.tools['libtriton']['sha256'],
            frontend_input_stage=runtime.frontend_input_stage),
            ascendc=dict(operator=OPERATORS['ascendc'],torch_npu_version=runtime.source_identity['torch_npu_version'],
                runtime_library_pins=cfg['runtime_library_pins']))
        state['arm_identities']=identities;save()
        for index in range(1 if args.calibration_only else 6):
            order=list(ARMS if index%2==0 else ARMS[::-1])
            row=dict(round_index=index,order=order,case=copy.deepcopy(state['case']),
                     measurement_mode='hardware-operation-duration',timing_unit='us',
                     input_fingerprint=inputs,arm_identities=identities)
            for arm in order:
                guard()
                block_path=args.output/('round-%02d'%index)/arm
                block=dict(round_index=index,arm=arm,path=str(block_path),finished=False)
                state['blocks'].append(block);state['phase']='round-%d-%s'%(index,arm);save()
                call=(lambda:runtime.ours(q,k,v,preset,bn)) if arm=='cvpipeline' else (lambda:runtime.ascendc(q,k,v))
                def initialized(_):
                    if arm=='cvpipeline':
                        compiled=runtime.compiles[(shape,preset,bn)]
                        if compiled['native_sha256'] != expected['native_sha256']:
                            raise ValueError('Fresh normal native image differs from its qualified image')
                latest,recipe=profile_block(torch,runtime.torch_npu,call,block_path,guard,initialized=initialized)
                pair=latest if arm=='cvpipeline' else (latest[0],runtime.lse_from_ascendc(latest,shape))
                block['output_checks']=[runtime.compare(pair[0],golden[0],arm+'-post-block-O'),runtime.compare(pair[1],golden[1],arm+'-post-block-M')]
                if not all(x['passed'] and x['finite'] for x in block['output_checks']):
                    raise ValueError('Post-timing full O/M check failed')
                after=[runtime.fingerprint(x) for x in (q,k,v)]
                if after!=inputs:
                    raise ValueError('Timing block changed Q/K/V inputs')
                core=args.cvp_core_type if arm=='cvpipeline' else args.ascendc_core_type
                csv_device=(infer_csv_device(block_path,arm=arm,core_type=core)
                            if args.calibration_only else args.csv_device_id)
                measured=select_block(block_path,device_id=csv_device,arm=arm,core_type=core)
                row[arm+'_op_statistic']=measured
                block.update(finished=True,recipe=recipe,measurement=measured,inputs_after=after)
                save();latest=None;pair=None
            state['rounds'].append(row);save()
        if args.calibration_only:
            ids={row['measurement']['device_id'] for row in state['blocks']}
            if len(ids)!=1:
                raise ValueError('Reference and candidate report different CSV device mappings')
            state['device_mapping']=dict(physical_npu=cfg['physical_npu'],logical_device=0,
                visible_devices=runtime.source_identity['physical_visibility'],csv_device_id=ids.pop(),
                evidence=[row['measurement'] for row in state['blocks']],
                excluded_from_performance_adoption=True)
            assert runtime.forward_counts=={preset:14,'ascendc':15},runtime.forward_counts
        else:
            state['summary']=summarize_paired_rounds(state['case'],state['rounds'])
            assert runtime.forward_counts=={preset:84,'ascendc':85},runtime.forward_counts
        state['attention_forward_counts']=runtime.forward_counts
        state['loaded_libraries']=runtime.loaded_libraries()
        state.update(exit=0,phase='device-mapping-calibration-only' if args.calibration_only else 'six-paired-hardware-rounds-complete')
    except Exception:
        state.update(exit=1,phase='failed',error=traceback.format_exc())
    finally:
        state.update(finished=True,finished_at=datetime.now(timezone.utc).isoformat());save()
    print(json.dumps({k:state.get(k) for k in ('case_id','phase','exit','summary','error')}),flush=True)
    return state['exit']


if __name__=='__main__':
    raise SystemExit(main())
