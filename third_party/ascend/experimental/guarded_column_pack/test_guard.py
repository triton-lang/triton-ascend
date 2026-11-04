import unittest,random,json,time
from dataclasses import replace
from pathlib import Path
from access_guard import *

def fixture(D=257,N=3,W=4,BD=256,itemsize=2):
    dtype='torch.float16' if itemsize==2 else 'torch.float32'
    lengths={'x':N*D,'cache':N*D*W,'y':N*D,'weight':D*W,'bias':D}
    start=0x10000;views={}
    for name,n in lengths.items():
        views[name]=StorageView(start,n*itemsize,start,itemsize,'npu:0',dtype)
        start+=n*itemsize+0x10000
    args=dict(D=D,N=N,W=W,BD=BD,grid=((D+BD-1)//BD,N),stride_x_n=D,stride_y_n=D)
    return views,args

class GuardTests(unittest.TestCase):
    def test_original_shapes(self):
        for N,D,s in [(16,2048,4),(32,2048,4),(16,4096,4),(32,4096,2),(64,2048,2),(16,6144,2),(32,6144,2),(64,4096,2),(128,2048,2),(64,6144,2)]:
            v,a=fixture(D,N,itemsize=s);self.assertTrue(conv_certificate_check(v,**a).fast)
    def test_tails(self):
        for D in (1,3,127,257,6145):
            v,a=fixture(D);self.assertTrue(conv_certificate_check(v,**a).fast)
    def test_short_allocation(self):
        v,a=fixture();v['cache']=replace(v['cache'],allocation_bytes=v['cache'].allocation_bytes-2)
        self.assertFalse(conv_certificate_check(v,**a).fast)
    def test_offset_view(self):
        v,a=fixture();v['cache']=replace(v['cache'],pointer=v['cache'].pointer+2)
        self.assertFalse(conv_certificate_check(v,**a).fast)
        v['cache']=replace(v['cache'],allocation_bytes=v['cache'].allocation_bytes+2)
        self.assertTrue(conv_certificate_check(v,**a).fast)
    def test_before_allocation(self):
        v,a=fixture();v['cache']=replace(v['cache'],pointer=v['cache'].pointer-2)
        self.assertFalse(conv_certificate_check(v,**a).fast)
    def test_aliases(self):
        for name in ('weight','bias','x','y'):
            v,a=fixture();v[name]=replace(v['cache'],pointer=v['cache'].pointer+2)
            self.assertFalse(conv_certificate_check(v,**a).fast)
    def test_disjoint_views(self):
        v,a=fixture();end=max(x.allocation_start+x.allocation_bytes for x in v.values());start=min(x.allocation_start for x in v.values())
        v={k:replace(x,allocation_start=start,allocation_bytes=end-start) for k,x in v.items()}
        self.assertTrue(conv_certificate_check(v,**a).fast)
    def test_readonly_alias(self):
        v,a=fixture();v['weight']=v['x'];self.assertTrue(conv_certificate_check(v,**a).fast is False) # x buffer too short for weight
        v['x']=v['cache'];self.assertFalse(conv_certificate_check(v,**a).fast)
    def test_partition(self):
        v,a=fixture();a['grid']=(1,3);self.assertFalse(conv_certificate_check(v,**a).fast)
        v,a=fixture();a['chunk_offset']=1;self.assertFalse(conv_certificate_check(v,**a).fast)
    def test_stride(self):
        v,a=fixture();a['stride_y_n']=0;self.assertFalse(conv_certificate_check(v,**a).fast)
    def test_overflow(self):
        v,a=fixture(D=(1<<30),N=1,BD=1<<30);self.assertFalse(conv_certificate_check(v,**a).fast)
    def test_device_dtype(self):
        for kw in ({'device':'npu:1'},{'dtype':'torch.bfloat16'}):
            v,a=fixture();v['weight']=replace(v['weight'],**kw);self.assertFalse(conv_certificate_check(v,**a).fast)
    def test_decisions_not_cached(self):
        v,a=fixture();self.assertTrue(conv_certificate_check(v,**a).fast)
        v['weight']=v['cache'];self.assertFalse(conv_certificate_check(v,**a).fast)
    def test_random_envelopes(self):
        rng=random.Random(42)
        for _ in range(10000):
            length=rng.randint(1,300);ptr=rng.randint(-10,300);rows=rng.randint(1,10);cols=rng.randint(1,10);stride=rng.randint(cols,12)
            v=StorageView(4096,length*2,4096+ptr*2,2,'npu:0','torch.float16')
            ok=check_envelopes({'p':v},(Footprint('p',rows,cols,stride),)).fast
            exact=all(0<=ptr+i*stride+j<length for i in range(rows) for j in range(cols))
            self.assertEqual(ok,exact)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(GuardTests);result=unittest.TextTestRunner(verbosity=2).run(suite)
    Path(__file__).with_name('unit_result.json').write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'device_launches':0},indent=2))
    raise SystemExit(not result.wasSuccessful())
