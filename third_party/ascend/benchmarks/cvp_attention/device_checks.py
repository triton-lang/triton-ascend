"""Full-element checks on device; independent mathematical references remain CPU."""
import hashlib

ATOL=RTOL=0.005
CHUNK=4*1024*1024

def compare(torch,actual,reference,label,guard=lambda:None,chunk=CHUNK):
    assert actual.device.type==reference.device.type=='npu'
    assert tuple(actual.shape)==tuple(reference.shape)
    a=actual.detach().reshape(-1);b=reference.detach().reshape(-1)
    finite=True;bad=0;maximum=0.0;count=0
    for start in range(0,a.numel(),chunk):
        guard()
        x=a[start:start+chunk].float();y=b[start:start+chunk].float()
        assert x.device.type==y.device.type=='npu'
        finite &= bool(torch.isfinite(x).all().item() and torch.isfinite(y).all().item())
        delta=(x-y).abs()
        maximum=max(maximum,float(delta.max().item()))
        bad+=int((delta>ATOL+RTOL*y.abs()).sum().item())
        count+=x.numel()
    return {'label':label,'passed':finite and bad==0,'finite':finite,'bad_elements':bad,
            'elements':count,'max_abs_error':maximum,'atol':ATOL,'rtol':RTOL,'checker':'device-fp32-chunked'}

def fingerprint(torch,tensor,guard=lambda:None,chunk=CHUNK):
    assert tensor.device.type=='npu'
    flat=tensor.detach().view(-1);digest=hashlib.sha256();finite=True
    for start in range(0,flat.numel(),chunk):
        guard();part=flat[start:start+chunk]
        finite &= bool(torch.isfinite(part).all().item())
        cpu=part.cpu();digest.update(memoryview(cpu.numpy()).cast('B'))
    return {'sha256':digest.hexdigest(),'shape':list(tensor.shape),'dtype':str(tensor.dtype),
            'strides':list(tensor.stride()),'finite':finite}

def assert_equivalent(actual,expected):
    for key in ['passed','finite','bad_elements','elements','max_abs_error','atol','rtol']:
        assert actual[key]==expected[key],(key,actual[key],expected[key])

def validate(runtime,cpu_compare,cpu_fingerprint,guard=lambda:None):
    torch=runtime.torch;cases=[]
    generator=torch.Generator(device='cpu').manual_seed(43059)
    for dtype in [torch.float16,torch.float32]:
        reference=torch.randn((2,513),generator=generator,dtype=torch.float32).to(dtype)
        actual=reference.clone();actual.reshape(-1)[::31]+=0.02
        cases.append(('random-'+str(dtype),actual,reference))
        values=torch.tensor([0.,-0.,1.,-1.,float('nan'),float('inf'),float('-inf')],dtype=dtype)
        cases.append(('nonfinite-'+str(dtype),values,values.clone()))
        cases.append(('zeros-'+str(dtype),torch.zeros((2,513),dtype=dtype),torch.zeros((2,513),dtype=dtype)))
    reference=torch.tensor([0.,0.25,1.,-1.,100.,-100.],dtype=torch.float32)
    boundary=reference+(ATOL+RTOL*reference.abs())
    actual=torch.cat([boundary,torch.nextafter(boundary,torch.full_like(boundary,float('inf'))),
                      torch.nextafter(boundary,torch.full_like(boundary,float('-inf')))])
    cases.append(('threshold-adjacent-fp32',actual,reference.repeat(3)))
    half=boundary.half()
    cases.append(('threshold-adjacent-fp16',torch.cat([half,
                  torch.nextafter(half,torch.full_like(half,float('inf'))),
                  torch.nextafter(half,torch.full_like(half,float('-inf')))]),reference.half().repeat(3)))
    reference=torch.linspace(-1.,1.,CHUNK+17,dtype=torch.float32)
    actual=reference.clone();actual[[0,CHUNK-1,CHUNK,CHUNK+16]]+=0.02
    cases.append(('chunk-boundary',actual,reference))
    results=[]
    for name,a,b in cases:
        guard();device_a=a.npu();device_b=b.npu()
        got=compare(torch,device_a,device_b,name,guard)
        expected=cpu_compare(device_a,device_b,name)
        assert_equivalent(got,expected)
        got_fingerprint=fingerprint(torch,device_a,guard)
        expected_fingerprint=cpu_fingerprint(device_a)
        assert got_fingerprint==expected_fingerprint,(name,'fingerprint differs')
        results.append({'case':name,'comparison_agrees':True,'fingerprint_agrees':True,
                        'elements':got['elements'],'finite':got['finite'],'expected_numeric_pass':expected['passed'],
                        'bad_elements':got['bad_elements'],'max_abs_error':got['max_abs_error']})
        del device_a,device_b
    return {'passed':True,'cases':results,'comparison_tolerances':{'atol':ATOL,'rtol':RTOL},
            'CPU_reference_unchanged':True,'all_elements_checked':True}
