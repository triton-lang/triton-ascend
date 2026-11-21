"""Synthetic config/provenance tests; no hardware execution or timing evidence."""
from pathlib import Path
import copy
import hashlib
import json
import tempfile
import unittest

from paired_stats import PRODUCTION_SHAPES
from run_host_comparison import validate_qualification


class HostProfileQualificationTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.cases=[dict(case_id=f'n{s[2]}-hd{s[3]}',shape=list(s),block_m=128,
                         block_n=256 if s[3]==64 else 128) for s in PRODUCTION_SHAPES]
        self.cfg=dict(cases=self.cases,measurement_mode='hardware-operation-duration',
            physical_npu=1,wheel_sha256='synthetic-wheel',frontend_sha256='synthetic-frontend',
            expected_boot_id='synthetic-boot',qualified_results={})
        qcfg={k:self.cfg[k] for k in ('physical_npu','wheel_sha256','frontend_sha256')}
        self.cfg['qualification_config_path'],qsha=self.write('qualification-config.json',qcfg)
        records=[]
        for case in self.cases:
            path,digest=self.write(case['case_id']+'.json',dict(finished=True,exit=0))
            records.append(dict(case_id=case['case_id'],shape=case['shape'],actual_exit=0,result_exit=0,
                                result_path=path,result_sha256=digest))
            self.cfg['qualified_results'][case['case_id']]=dict(path=path,sha256=digest)
        records += [dict(case_id=f'small-{i}',shape=[1,1,512,64],actual_exit=0,result_exit=0)
                    for i in range(14)]
        self.host=dict(finished=True,exit=0,admission_valid=True,config_sha256=qsha,cases=records)
        self.seal_host()
        self.mapping=dict(finished=True,exit=0,calibration_only=True,boot_id='synthetic-boot',
            device_mapping=dict(physical_npu=1,logical_device=0,visible_devices='1',csv_device_id=0,
                                excluded_from_performance_adoption=True))
        self.seal_mapping()

    def write(self,name,value):
        path=self.root/name
        data=(json.dumps(value,indent=2)+'\n').encode()
        path.write_bytes(data)
        return str(path),hashlib.sha256(data).hexdigest()

    def seal_host(self):
        path,digest=self.write('qualification-host.json',self.host)
        self.cfg.update(qualification_host_path=path,qualification_host_sha256=digest)

    def seal_mapping(self):
        path,digest=self.write('mapping.json',self.mapping)
        self.cfg.update(device_mapping_path=path,device_mapping_sha256=digest)

    def test_complete_scope_and_physical_one_csv_zero_are_valid(self):
        self.assertEqual(validate_qualification(self.cfg),0)

    def test_mapping_calibration_needs_one_qualified_case_and_no_prior_mapping(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['measurement_mode']='mapping-calibration';cfg['cases']=cfg['cases'][:1]
        cfg['qualified_results']={cfg['cases'][0]['case_id']:cfg['qualified_results'][cfg['cases'][0]['case_id']]}
        cfg.pop('device_mapping_path');cfg.pop('device_mapping_sha256')
        self.assertIsNone(validate_qualification(cfg))

    def test_partial_failed_or_unadmitted_qualification_rejects(self):
        for key,value in [('finished',False),('exit',1),('admission_valid',False)]:
            with self.subTest(key=key):
                original=self.host[key];self.host[key]=value;self.seal_host()
                with self.assertRaises(ValueError):validate_qualification(self.cfg)
                self.host[key]=original
        self.seal_host()
        self.host['cases'].pop();self.seal_host()
        with self.assertRaises(ValueError):validate_qualification(self.cfg)

    def test_duplicate_or_failed_numerical_case_rejects(self):
        self.host['cases'][0]['actual_exit']=1;self.seal_host()
        with self.assertRaises(ValueError):validate_qualification(self.cfg)
        self.host['cases'][0]['actual_exit']=0
        self.host['cases'][-1]['case_id']=self.host['cases'][0]['case_id'];self.seal_host()
        with self.assertRaises(ValueError):validate_qualification(self.cfg)

    def test_changed_binary_or_physical_device_rejects(self):
        for key,value in [('wheel_sha256','other'),('frontend_sha256','other'),('physical_npu',2)]:
            with self.subTest(key=key):
                cfg=dict(self.cfg,**{key:value})
                with self.assertRaises(ValueError):validate_qualification(cfg)

    def test_changed_result_bytes_or_inventory_rejects(self):
        proof=self.cfg['qualified_results'][self.cases[0]['case_id']]
        Path(proof['path']).write_text('{}')
        with self.assertRaises(ValueError):validate_qualification(self.cfg)
        cfg=copy.deepcopy(self.cfg);cfg['qualified_results'].pop(self.cases[-1]['case_id'])
        with self.assertRaises(ValueError):validate_qualification(cfg)

    def test_reordered_duplicate_or_wrong_tiles_reject(self):
        for cases in (list(reversed(self.cases)),self.cases+[self.cases[0]],
                      [dict(self.cases[0],block_m=256)]+self.cases[1:]):
            with self.subTest(cases=cases[:1]):
                with self.assertRaises(ValueError):validate_qualification(dict(self.cfg,cases=cases))

    def test_untrusted_stale_or_wrong_device_mapping_rejects(self):
        original=copy.deepcopy(self.mapping)
        for key,value in [('physical_npu',2),('logical_device',1),('visible_devices','2'),
                          ('csv_device_id',-1),('csv_device_id',True),
                          ('excluded_from_performance_adoption',False)]:
            with self.subTest(key=key,value=value):
                self.mapping=copy.deepcopy(original);self.mapping['device_mapping'][key]=value
                self.seal_mapping()
                with self.assertRaises(ValueError):validate_qualification(self.cfg)
        self.mapping=copy.deepcopy(original);self.mapping['boot_id']='other';self.seal_mapping()
        with self.assertRaises(ValueError):validate_qualification(self.cfg)


if __name__=='__main__':
    unittest.main(verbosity=2)
