"""Physical-host admission controller for this fork's serial paired profiling cases."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,re,shutil,subprocess,sys,time,traceback

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save_json(path,value):
    temporary=Path(path).with_name(Path(path).name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def identity(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    ns=re.search(r'^NSpid:\s*(.*)$',(p/'status').read_text(),re.M)
    return {'pid':pid,'start_ticks':int(fields[19]),'parent_pid':int(fields[1]),
            'namespace_pids':list(map(int,ns[1].split())) if ns else [pid]}

def parse_device_snapshot(text,physical):
    health=re.search(r'^\|\s*'+str(physical)+r'\s*\|\s*(Ascend\S*)\s*\|\s*(\S+)',text,re.M)
    assert health,'Selected physical NPU health cannot be parsed'
    assert 'Process id' in text,'NPU process table is missing'
    section=text.split('Process id',1)[1]
    pids=sorted({int(m[1]) for m in re.finditer(r'^\|\s*'+str(physical)+r'\s*\|\s*(\d+)\s*\|',section,re.M)})
    assert pids or 'No running processes found in NPU '+str(physical) in section,'Selected device process state is unknown'
    return health[2],pids

def checked_json(path, expected):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected:
        raise ValueError('Qualification/mapping checksum differs: '+str(path))
    return json.loads(data)


def validate_qualification(cfg):
    from paired_stats import PRODUCTION_SHAPES
    cases=cfg['cases']
    shapes=[tuple(c['shape']) for c in cases]
    calibration=cfg['measurement_mode']=='mapping-calibration'
    if len({c['case_id'] for c in cases})!=len(cases):
        raise ValueError('Duplicate selected case')
    if calibration:
        if len(cases)!=1 or shapes[0] not in PRODUCTION_SHAPES:
            raise ValueError('Mapping calibration needs one production case')
    elif shapes!=list(PRODUCTION_SHAPES):
        raise ValueError('Primary performance cohort needs all eleven ordered shapes')
    if any(c['block_m']!=128 or c['block_n'] not in (128,256) for c in cases):
        raise ValueError('Unexpected tile geometry')
    host=checked_json(cfg['qualification_host_path'],cfg['qualification_host_sha256'])
    if not (host.get('finished') is True and host.get('exit')==0
            and host.get('admission_valid') is True and len(host.get('cases',[]))==25):
        raise ValueError('Complete admitted25-case qualification is required')
    qualification=checked_json(cfg['qualification_config_path'],host['config_sha256'])
    for key in ('wheel_sha256','frontend_sha256','physical_npu'):
        if qualification.get(key)!=cfg.get(key):
            raise ValueError('Qualification '+key+' differs')
    records={c['case_id']:c for c in host['cases']}
    if len(records)!=25 or any(c.get('actual_exit')!=0 or c.get('result_exit')!=0 for c in records.values()):
        raise ValueError('Qualification contains failed or duplicate cases')
    if set(cfg['qualified_results'])!={c['case_id'] for c in cases}:
        raise ValueError('Selected qualification inventory differs')
    for case in cases:
        proof=cfg['qualified_results'][case['case_id']]
        record=records.get(case['case_id'])
        if (not record or record['shape']!=case['shape'] or
                record['result_sha256']!=proof['sha256'] or record['result_path']!=proof['path']):
            raise ValueError('Selected qualification result differs')
        result=checked_json(proof['path'],proof['sha256'])
        if result.get('finished') is not True or result.get('exit')!=0:
            raise ValueError('Selected numerical result is not successful')
    if calibration:
        return None
    mapping=checked_json(cfg['device_mapping_path'],cfg['device_mapping_sha256'])
    device=mapping.get('device_mapping',{})
    if not (mapping.get('finished') is True and mapping.get('exit')==0
            and mapping.get('calibration_only') is True
            and device.get('excluded_from_performance_adoption') is True
            and device.get('physical_npu')==cfg['physical_npu']
            and mapping.get('boot_id')==cfg['expected_boot_id']
            and device.get('logical_device')==0
            and str(device.get('visible_devices'))==str(cfg['physical_npu'])
            and type(device.get('csv_device_id')) is int and device['csv_device_id']>=0):
        raise ValueError('Missing fresh physical/logical/CSV device mapping')
    return device['csv_device_id']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    ROOT=Path(cfg['workspace_root'])
    assert isinstance(cfg['container'],str) and cfg['container']
    assert not Path('/.dockerenv').exists()
    assert cfg['authorized'] and cfg['stage6_excluded'] and cfg['measurement_mode'] in ('mapping-calibration','hardware-operation-duration')
    assert cfg['experiment_id']=='20261104T072938Z-79d49a12b42a'
    assert sha(__file__)==cfg['host_script_sha256']
    run=Path(cfg['run_dir']);source=Path(cfg['source_root'])
    assert ROOT.resolve()==ROOT and run.resolve().is_relative_to(ROOT) and source.resolve().is_relative_to(ROOT)
    assert not run.exists()
    physical=cfg['physical_npu'];assert type(physical) is int and 0<=physical<=7
    assert sha(cfg['wheel_path'])==cfg['wheel_sha256']
    assert sha(cfg['kernel_path'])=='33faa144cbdde5bad8ec8a05bffab664130b9a285bec8985d5635f0cf38f0ac1'
    for name,digest in cfg['input_hashes'].items():
        path=source/name;assert path.resolve().is_relative_to(source) and sha(path)==digest,name
    run.mkdir();save_json(run/'CONFIG.json',cfg)
    state={'controller':identity(os.getpid()),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
           'job_id':cfg['job_id'],'nonce':cfg['nonce'],'config_sha256':sha(args.config),
           'started_at':datetime.now(timezone.utc).isoformat(),'phase':'preflight','finished':False,'exit':None,
           'physical_npu':physical,'cases':[],'observations':[],'monitor_gap_count':0,'admission_valid':True,
           'worker_host_identity':None,'profiled_active_tasks':0,'stage6_excluded':True,
           'sampling_caveat':'Discrete observations do not prove continuous occupancy; gaps are preserved and stale observations hold new launches.'}
    save=lambda:save_json(run/'status.json',state)
    stop=run/'STOP';case_id=None;result_path=None;reserved=False
    def ours(resident):
        worker=state['worker_host_identity'];seen=set();current=resident
        if not worker:return False
        while current['pid']>1 and current['pid'] not in seen:
            if current['pid']==worker['pid']:return current['start_ticks']==worker['start_ticks']
            seen.add(current['pid'])
            try:current=identity(current['parent_pid'])
            except (FileNotFoundError,ProcessLookupError):return False
        return False
    def sample():
        started=time.time()
        result=subprocess.run(['/usr/local/sbin/npu-smi','info'],capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
        text=result.stdout;health,pids=parse_device_snapshot(text,physical)
        residents=[identity(pid) for pid in pids]
        if result_path is not None and result_path.exists():
            worker=json.loads(result_path.read_text())
            for resident in residents:
                if resident['namespace_pids'][-1]==worker['pid'] and resident['start_ticks']==worker['start_ticks']:
                    assert worker['boot_id']==state['boot_id'];state['worker_host_identity']=resident
        own=[row for row in residents if ours(row)];external=[row for row in residents if not ours(row)]
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        row={'epoch':time.time(),'query_started_epoch':started,'case_id':case_id,'physical_npu':physical,
             'health':health,'resident_identities':residents,'own_resident_pids':[r['pid'] for r in own],
             'external_resident_pids':[r['pid'] for r in external],'external_workloads':len(external),
             'workload_count_including_reserved_own':len(external)+int(reserved or bool(own)),
             'load':list(os.getloadavg()),'high_load_authorized':True,'available_ram_bytes':available,
             'free_bytes':shutil.disk_usage(ROOT).free}
        eligible=(row['health']=='OK' and len(external)<=2 and row['workload_count_including_reserved_own']<=3
                  and available>=64*2**30 and row['free_bytes']>=40*2**30)
        row['query_seconds']=row['epoch']-started;row['allowed']=eligible and row['query_seconds']<=25
        row['status']='observed' if row['allowed'] else ('rejected' if not eligible else 'monitor_unavailable')
        (run/('npu-smi-'+str(len(state['observations']))+'.txt')).write_text(text)
        return row
    def observe(before=False):
        started=time.monotonic()
        while True:
            query_started=time.time()
            try:row=sample()
            except Exception as error:
                row={'epoch':time.time(),'query_started_epoch':query_started,'physical_npu':physical,
                     'allowed':False,'status':'monitor_unavailable','reason':str(error),'high_load_authorized':True}
            if row['status']=='monitor_unavailable':state['monitor_gap_count']+=1
            state['observations'].append(row);save_json(run/'ADMISSION.json',row);save()
            if row['status']=='rejected':
                state['admission_valid']=False
                if not stop.exists():stop.write_text('Actual admission rejection: '+json.dumps(row))
                if before:raise RuntimeError('Actual device/resource admission rejected')
            if not before or row['allowed']:return row
            if time.monotonic()-started>=180:raise RuntimeError('No fresh admitted device observation within180seconds')
            time.sleep(0.25)
    save()
    try:
        csv_device=validate_qualification(cfg)
        assert state['boot_id']==cfg['expected_boot_id']
        calibration=cfg['measurement_mode']=='mapping-calibration'
        work=[dict(case,independent_repeat=False) for case in cfg['cases']]
        for case in work:
            case_id=case.get('output_id',case['case_id']);state['phase']=case_id
            assert not stop.exists()
            reserved=False;state['worker_host_identity']=None;result_path=None;observe(before=True)
            folder=run/case_id;folder.mkdir();(folder/'tmp').mkdir()
            result_path=folder/'run/RESULT.json';reserved=True
            env=dict(cfg['environment'])
            env.update(PYTHONPATH=cfg['wheel_site']+':'+str(source),ASCEND_RT_VISIBLE_DEVICES=str(physical),
                TRITON_ALWAYS_COMPILE='0',TRITON_HOME=str(folder/'triton-home'),TRITON_CACHE_DIR=str(folder/'cache'),
                TRITON_DEBUG='0',TRITON_KERNEL_DUMP='0',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
                TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'xdg'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            assert not any('B111' in str(v) for v in env.values())
            for name in ['TRITON_OVERRIDE_DIR','TRITON_KERNEL_OVERRIDE','LLVM_IR_ENABLE_DUMP','MLIR_ENABLE_DUMP']:
                assert not env.get(name);env[name]=''
            row=observe(before=True);high=row['load'][0]>700
            prefix=['nice','-n','10' if high else '0','ionice','-c','3' if high else '2']
            if not high:prefix+=['-n','4']
            env.update(MAX_JOBS='4' if high else '16',CMAKE_BUILD_PARALLEL_LEVEL='4' if high else '16',TRITON_PARALLEL_LINK_JOBS='1')
            argv=['docker','exec',*[value for key,item in env.items() for value in ('-e',key+'='+str(item))],
                  cfg['container'],*prefix,cfg['python_path'],'-B',str(source/'run_comparison.py'),
                  '--config',str(run/'CONFIG.json'),'--recipes',cfg['recipes_path'],'--case-id',case['case_id'],
                  '--qualified-result',cfg['qualified_results'][case['case_id']]['path'],
                  '--qualified-sha256',cfg['qualified_results'][case['case_id']]['sha256'],
                  '--output',str(folder/'run'),'--stop-marker',str(stop),'--admission',str(run/'ADMISSION.json'),
                  '--cvp-core-type',cfg['cvp_core_type'],'--ascendc-core-type',cfg['ascendc_core_type']]
            argv+=['--calibration-only'] if calibration else ['--csv-device-id',str(csv_device)]
            record={'case_id':case_id,'source_case_id':case['case_id'],'shape':case['shape'],'independent_repeat':case['independent_repeat'],'argv':argv,'environment':env}
            state['cases'].append(record)
            with (folder/'worker.stdout').open('xb') as output,(folder/'worker.stderr').open('xb') as errors:
                child=subprocess.Popen(argv,cwd=source,stdin=subprocess.DEVNULL,stdout=output,stderr=errors)
                record['docker_exec_identity']=identity(child.pid);save()
                while child.poll() is None:
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:observe()
                record['actual_exit']=child.returncode
            reserved=False;observe(before=True)
            result=json.loads(result_path.read_text()) if result_path.exists() else None
            record.update(result_path=str(result_path),result_sha256=sha(result_path) if result else None,
                          result_exit=result.get('exit') if result else None,worker_host_identity=state['worker_host_identity'])
            save()
            assert child.returncode==0 and state['admission_valid'] and result and result['exit']==0,record
            assert result['finished']
            assert state['worker_host_identity'] is not None,'Physical NPU process identity was not observed'
            assert result['calibration_only']==calibration
            expected_blocks=2 if calibration else 12
            assert len(result['blocks'])==expected_blocks
            record['profiled_active_tasks']=10*expected_blocks
            record['collector_phase']=result['phase']
            if not calibration:
                record['summary']=result['summary']
                if result['summary']['independent_repeat_needed'] and not case['independent_repeat']:
                    work.append(dict(case,output_id=case['case_id']+'-independent-repeat-1',independent_repeat=True))
            save()
        primary=[x for x in state['cases'] if not x['independent_repeat']]
        assert len(primary)==(1 if calibration else 11)
        state['profiled_active_tasks']=sum(x['profiled_active_tasks'] for x in state['cases'])
        state['independent_repeat_cases']=[x['source_case_id'] for x in state['cases'] if x['independent_repeat']]
        state['no_cohort_pooling']=True
        state['timing_admission_label']='Discrete host observations; sharing withinthreeworkloads is allowed; inspect raw observations and preserved gaps'
        state.update(exit=0,phase='device-mapping-calibration-complete' if calibration else 'all11-paired-timing-cohorts-complete')
    except Exception:
        if not stop.exists():stop.write_text('Stop new launches; preserve all evidence.')
        state.update(exit=1,phase='failed',error=traceback.format_exc())
    finally:
        state.update(finished=True,finished_at=datetime.now(timezone.utc).isoformat());save()
        print(json.dumps({k:state.get(k) for k in ['phase','exit','admission_valid','error']}),flush=True)
    return state['exit']
if __name__=='__main__':sys.exit(main())
