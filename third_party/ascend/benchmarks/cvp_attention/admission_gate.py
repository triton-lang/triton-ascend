"""Hold new launches until a fresh, valid admission observation is available."""
from pathlib import Path
import json,math,time

class AdmissionStopped(RuntimeError):
    pass

class AdmissionUnavailable(RuntimeError):
    pass

def evaluate(row,physical,now,freshness_seconds=25):
    if not isinstance(row,dict):return 'wait','unreadable observation'
    if row.get('physical_npu') not in [None,physical]:return 'stop','wrong physical device'
    if row.get('health') not in [None,'OK']:return 'stop','device health rejected'
    limits=[('external_workloads',2,False),('workload_count_including_reserved_own',3,False),
            ('available_ram_bytes',64*1024**3,True),('free_bytes',40*1024**3,True)]
    for key,limit,minimum in limits:
        value=row.get(key)
        if value is not None and (value<limit if minimum else value>limit):
            return 'stop',key+' rejected'
    if row.get('status')=='rejected':return 'stop','host rejected admission'
    if row.get('allowed') is not True:return 'wait',row.get('reason','observation unavailable')
    required=['physical_npu','health','external_workloads','workload_count_including_reserved_own','available_ram_bytes','free_bytes']
    if any(row.get(key) is None for key in required):return 'wait','incomplete observation'
    if row['workload_count_including_reserved_own']!=row['external_workloads']+1:
        return 'stop','own workload reservation missing'
    if not row.get('high_load_authorized'):return 'stop','high-load policy mismatch'
    epoch=row.get('query_started_epoch',row.get('epoch'))
    if not isinstance(epoch,(int,float)) or not math.isfinite(epoch):return 'wait','invalid observation time'
    age=now-epoch
    if not 0<=age<=freshness_seconds:return 'wait','stale observation'
    return 'allow','fresh valid observation'

class AdmissionGate:
    def __init__(self,admission,stop_marker,physical,on_wait=None,clock=time.time,
                 monotonic=time.monotonic,sleep=time.sleep,recovery_seconds=180):
        self.admission=Path(admission);self.stop=Path(stop_marker);self.physical=physical
        self.on_wait=on_wait;self.clock=clock;self.monotonic=monotonic;self.sleep=sleep
        self.recovery_seconds=recovery_seconds

    def __call__(self):
        started=None;record=None
        def finish_wait(recovered,reason):
            if record is not None:
                record.update(finished_epoch=self.clock(),seconds=self.monotonic()-started,
                              recovered=recovered,end_reason=reason)
                if self.on_wait:self.on_wait(record)
        while True:
            if self.stop.exists():
                reason=self.stop.read_text()[:1000];finish_wait(False,reason)
                raise AdmissionStopped(reason)
            try:row=json.loads(self.admission.read_text())
            except (FileNotFoundError,json.JSONDecodeError):row=None
            decision,reason=evaluate(row,self.physical,self.clock())
            if decision=='stop':
                finish_wait(False,reason)
                raise AdmissionStopped(reason)
            if decision=='allow':
                if record is not None:
                    record['accepted_query_started_epoch']=row.get('query_started_epoch',row['epoch'])
                    finish_wait(True,reason)
                return
            if started is None:
                started=self.monotonic();record={'started_epoch':self.clock(),'reasons':[],'recovered':False}
            if reason not in record['reasons']:record['reasons'].append(reason)
            if self.monotonic()-started>=self.recovery_seconds:
                finish_wait(False,'recovery timeout')
                raise AdmissionUnavailable('No fresh valid admission within recovery limit')
            self.sleep(0.25)
