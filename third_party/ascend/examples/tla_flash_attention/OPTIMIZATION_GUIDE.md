# Source-guided FlashAttention optimization on Ascend950

## 1. Scope and result

This study implements exact-workload BF16 attention with TLA on Ascend950PR,
using the installed AscendC `npu_fusion_attention` implementation as the native
reference. It starts from the work associated with
[AscendNPU-IR issue 104](https://gitcode.com/liupengcheng2012/AscendNPU-IR/issues/104).
The final selection improves five of the eight previously optimized TLA cases;
FA7 is 4.3% faster than AscendC in the retained paired run. The other seven cases
remain slower. This is a finite, shared-device study, not a global optimum.

[PERFORMANCE.md](PERFORMANCE.md) contains every case, configuration, numerical
criterion and uncertainty. [kernel.py](kernel.py) is the readable implementation;
[run.py](run.py) reproduces correctness and kernel-only profiling.

## 2. Freeze the comparison contract first

Use the same inputs, output contract, precision policy, scale, masking, dropout
and actual core counts. Allow each implementation its own internal tiling.
For this study the contract is:

- B=4, H=KVH=8, contiguous BNSD BF16 Q/K/V; S=1024 or 8192, D=64 or 128.
- Dense or ordinary causal attention; scale is D**-0.5, no dropout, padding or PSE.
- BF16 probability operands and output; FP32 matrix accumulation, online-softmax
  state, exponent/subtraction/division and output accumulation. No HF32/TF32.
- Output O plus **all eight** FP32 replicas of row maximum and exponential sum.
- Native `inner_precise=0`, `keep_prob=1`, `sparse_mode=0` for dense or 2 with
  a compressed 2048x2048 upper-triangular Boolean mask for causal attention.
- Every measured launch has 28 AIC and 56 AIV, checked from profiler metadata.

A compiler flag or API name alone does not prove numerical equivalence. Inspect
native dispatch, source types, and final IR. A different reduction grouping is
allowed by different tiling; lowering precision or dropping valid keys is not.
The output-error thresholds in PERFORMANCE.md are empirical acceptance criteria,
not an asserted official API guarantee.

## 3. Establish an evidence chain before changing the kernel

Use this sequence for other kernels:

1. Record the native API call and actual selected specialization for **each**
   workload. Decode template-key fields using that installed source version.
2. Read the selected source branches, including layout and buffer policies.
   Do not assume a representative case covers all dispatches.
3. Inspect TLA's frontend IR and the IR immediately before backend lowering.
   Trace logical shape, physical layout, operand transpose flags and real extents.
4. Inspect the final template/library calls or decoded instruction trace.
   A missing Python interface does not imply missing hardware/compiler support.
5. Form one falsifiable hypothesis, change kernel structure, validate all outputs,
   then measure the paired device kernel. Keep rejected results outside the final
   runnable package, but document what they ruled out.

Here, ordinary objdump recognized the ELF but could not decode the opcodes.
Simulator instruction logs supplied the useful decoded instructions. Source file identities are listed in [source_references.json](evidence/source_references.json);
compact decoded VF data is in [simulator_summary.json](evidence/simulator_summary.json).
The installed sources inspected include `flash_attention_score_common_regbase.h`,
`flash_attention_noquant_block_cube.h` and the native template-key header.
The compiler transpose path is `FoldVtransposePattern` in `NormalizeMatmul.cpp`,
MMAD library naming in `LibraryFunctionOpInterfaceImpl.cpp`, and the transpose-aware
`RegBase/Cube/LocalMmad.cpp` template. Hardware
counter collection failed, so zero/missing counter values were not evidence.
The simulator used a different model (9599 versus the requested 9579), and native
execution was incomplete. Those traces support instruction-level hypotheses,
not whole-kernel latency or utilization claims.

## 4. Orient both matrix products around the vector layout

The efficient score path computes **K @ Q.T**, with physical score shape
[BN, BM]. NZ2ND Fixpipe with COLUMN_SPLIT sends each vector sub-block [BN, 64].
Each FP32 SIMD lane then owns one query row, and traverses keys vertically.

This avoids repeated horizontal reductions across query/key vectors. Four
independent maximum accumulators scan adjacent key rows; four independent sum
accumulators process the exponential pass. The final reductions combine these
four vectors without changing FP32 arithmetic.

The packed probability buffer has physical NZ orientation corresponding to
[BN, BM]. PV uses `transpose_a=True`, computing P.T @ V to obtain [BM, D].
**Transpose is already absorbed into MMAD's `_ta` template.** The performance
problem was not an unavoidable vector transpose: it was expressing a useful
physical layout and exposing the existing transpose-A frontend interface.
Inspect transpose folding before paying for a layout-conversion optimization.

## 5. Match native operations and packing without reducing precision

`vexpdif(x, m)` exposes the existing FP32 AVE ExpSub instruction directly.
`vmuls(x, scale)` uses a scalar multiplication instruction instead of requiring
a broadcast vector. These are native operation interfaces, not substitute
approximations. Inputs, accumulators and normalization precision remain aligned
with the native high-precision BF16 path.

The softmax packs four BF16 probability columns into two scatter-block stores
using deinterleave operations. A one-element padding term gives a 129-element
half-tile pitch for BN=256. The two copies into L1 reconstruct the logical P
layout needed by transposed-A MMAD. Every physical pitch and split must agree
between register packing, UB views, L1 views and matrix interpretation.

In the completed first softmax VF from partial traces, old TLA/native/new TLA
issued 2244/2074/2068 instructions, with issue spans 1807/1082/1044 cycles and VF
execution 2204/1593/1513 cycles. Native and new TLA both used 256 ExpSub operations,
257 scalar multiplies, 512 score loads and 128 scatter-block stores. The old
64 horizontal maxima and 64 horizontal sums disappeared. This explains a local
improvement; it does not establish end-to-end dominance or measure the later
PV double buffering.

## 6. Design the pipeline around buffer lifetime

Each logical task is a (query tile, key tile) pair. Successive tasks can cross
query boundaries without draining the entire pipeline. At logical time t:

| Engine | Work | Resource |
|---|---|---|
| Cube | QK(t) | Two score UB slots |
| Vector | softmax(t-1) | Three P/alpha slots; three query-state slots |
| Cube | PV(t-2) | One or two FP32 PV UB slots |
| Vector | output update(t-3) | One running FP32 output accumulator |

The static phase period is lcm(2,3)=6. The implementation expands these phases
before TLA compilation; each phase uses a statically selected physical buffer.
Shape-derived predicates handle prologue and epilogue tasks.

Buffer rotation alone does not establish correctness. For every resource record:

- The initial free credits.
- The producer's last write and publication event.
- The consumer's wait before its first read.
- The consumer's last read and release event.
- The producer's wait before reusing the slot, plus final drain obligations.

Score readiness uses flags 0/4 and score reuse 10/11. P readiness uses 1/5/13.
Double-PV uses readiness 2/6 and reuse 9/12. Both PV slots receive an initial
free credit, and both free credits are drained before exit. A producer must
never overwrite a tile whose consumer has only started, rather than completed,
its read.

D64 has enough UB capacity for two PV tiles. This removes unnecessary Cube/Vector
serialization and improves FA3/FA4/FA8. D128 with BN=256 cannot simply allocate
another full PV tile; reducing BN to 128 made double-PV variants feasible but
slower in screening. Buffer count is a capacity-and-dependency decision, not
an unconditional optimization. Long cases retain double K/V L1 buffering;
short cases use the simpler single K/V schedule selected by measurements.

## 7. Specialize structural tails consistently across the pipeline

For BM=128 and BN=256, a causal final tile can have exactly 128 contributing keys.
The compact path trims all relevant stages together:

1. QK uses actual M=128, with a physically matching compact L0C buffer.
2. Fixpipe and softmax touch only the compact score tile.
3. Probability packing and UB-to-L1 copies use the compact pitch.
4. PV uses actual K=128; the final output remains [BM,D].

Using a full L0C shape with smaller MMAD real M gives the wrong packed layout
for Fixpipe. Trimming only Cube or only Vector left much of the cost behind.
`prefix_view` aliases a bounded contiguous allocation prefix with a different
pitch; it does not allocate extra UB. It rejects nonzero/unknown origins and
non-contiguous sources because reinterpret_cast offsets refer to the base.
No causally valid key is omitted.

## 8. Balance causal work by public shape

Equal query counts can have very different causal key counts. Long cases use
head-dependent rotation. FA7 reverses query order in odd heads. FA5 uses a fixed
per-core assignment of all 256 query tiles, based on valid key-block count plus
a fixed query overhead of two. Greedy assignment followed by pairwise swaps
reduces squared estimated core loads. The resulting query IDs are embedded in
scalar constants, with every query occurring exactly once.

This is compile-time workload scheduling. There is no input-value inspection,
external output cache, runtime host preprocessing, omitted task or benchmark-only
path. Recheck bijection and complete key coverage whenever changing the mapping.

The compact paths also broadcast all eight max/sum replicas using `E2B_B32`
loads. This is bit-preserving output construction; it had no clear standalone
speed benefit and is not credited as the main gain.

## 9. Preserve a readable implementation without changing generated work

The original six selected files contained about 19,400 lines of manually
expanded phases and repeated softmax bodies. `kernel.py` names the shared QK,
softmax, PV and output stages. `expand_kernel.py` performs only:

- Syntactic substitution of named stage parameters.
- Selection of first/masked/tail softmax bodies at source-generation time.
- Six-way rotation of physical buffer names and their corresponding flags.

It does not reorder instructions or reassociate arithmetic. Six AST digests
match the measured source exactly, and **all eight launched binaries are
byte-identical** to the frozen performance artifacts. All four precision families
were rerun through the new public wrapper: 152 full-output checks per implementation.
Consequently the readability refactor preserves the measured device code and its
performance; the performance table is not replaced by a noisy new timing sample.

The `compat` path supports the measured wheel using typed existing AVE/HIVM
operations. `native` uses the accompanying direct frontend changes. Compatibility
adapters are isolated from the upstream library implementation and do not modify
shared site-packages. The direct C++ binding changes are not yet wheel-built;
54 source-loaded frontend contract tests pass. Do not transfer the performance
claim to a rebuilt wheel without code-generation and device validation.

## 10. Measure decisions, not noise

Compile and allocate outputs before timing. Profile actual device kernels;
Python dispatch/event timing is useful for screening but includes different
fixed overheads. Rotate order across multiple paired rounds, retain every sample,
verify launch core counts and cross-check trace durations with op-summary CSV.
Never pick only idle windows or discard unfavorable samples on a shared device.

Poison preallocated outputs before repeated correctness launches; test every
output replica against an independent FP32 reference. Include alternate random
inputs, uniform scores and peaked scores. Repeated launches of one seed check
repeatability, not independent coverage. Freeze shape-only selection before the
final paired run, and attach source/binary digests to the results.

## 11. Rejected paths and next experiments

Extra unrolling, a smaller probability pitch, BN=128 DN variants, partial-only
tail trimming and more elaborate D64 query mapping did not justify promotion.
Explicit L0 MMAD lowering failed in the tested installed conversion path; that
is a limitation of that path, not proof that every future pipeline lacks L0 support.

Priorities for further work:

1. **D128 PV overlap:** reduce live UB footprint or stream output updates enough
   to permit double buffering without the losing BN reduction. First measure
   PV-ready/free wait spans and peak live memory; then validate ownership edges.
2. **Per-case layout:** native short causal FA5/FA7 uses ND whereas the selected
   TLA path uses DN. Test a short-case ND specialization with matched math, then
   compare complete kernels rather than copying FA8's layout universally.
3. **FA3/FA8 remaining small gap:** inspect actual Cube/Vector overlap and tail
   drain. Their gaps are 2.7% and 4.2%; small isolated VF wins may disappear in
   full-operation timing.
4. **Compiler interfaces:** build/test direct transpose-A, bounded real extents,
   prefix aliases and FP32 native SIMD interfaces. Explore explicit L0 interfaces
   only after proving supported lowering and resource lifetimes.
5. **Better traces:** obtain a complete, model-matched simulator replay or valid
   hardware counters. Avoid inferring steady-state utilization from partial VFs.

For every experiment: state the bottleneck, specify the falsifying measurement,
keep the numerical contract fixed, check complete output coverage, and promote
only on paired full-kernel measurements.
