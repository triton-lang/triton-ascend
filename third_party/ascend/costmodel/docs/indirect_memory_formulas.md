# Indirect Load and Store formulas

The compiler replaces one verified memory target contribution and preserves
auxiliary address, index, mask and other memory work. Each selected formula has
at most eight terms. Routing uses visible geometry, never case names or hidden
index contents. Readable coefficients remain in `OrdinaryLoadProgram.inc` and
`OrdinaryStoreSelectedFormulas.inc`; the latter also binds the Store model SHAs.

## Contracts and fallback

The generic measured domain covers rank 2–5 mixed S/D layouts ending in D,
SIMD/SIMT, F=1, static rectangular masks or no mask, and Known/Host/Device
strides. These results do not establish acceptance for the other 26 tail-S
layouts, page/multiple-base structures, data-dependent masks or arbitrary
kernels. Dedicated-pattern models are separate work.

Facts come from actual post-transform TTIR. Host strides require matching
entry-argument attributes and scalar bindings. Device strides remain opaque
and require a positive/no-wrap contract. Logical FP8 and byte transport remain
distinct. Unsupported types, geometry, dynamic masks, F>1, nested ownership and
multiple matched Load targets retain the existing compiler fallback.

Load is enabled by default; `ordinary_load_selected_enabled=False` disables it.
Store requires its explicit native ownership, first-store, IR/model identity,
visibility, lifecycle and matched-increment accounting contract. No standalone
Python predictor, owned-session, event or launcher API is added.

## Validation

| Model | Overall MAPE | SIMD | SIMT | Validation population |
| --- | ---: | ---: | ---: | --- |
| Load | 19.9795% | 21.6531% | 18.8747% | 1187 positive inputs; 29 nonpositive inputs retained separately |
| Store | 8.2126% | 5.3365% | 11.0887% | 256 configurations, 2048 input identities, two observation rounds |

Load uses a fixed 734 TRAIN / 152 development configuration split. Its 108
physical routes select 48 TRAIN short formulas, with 62 formulas including base
fallbacks. Development data selected existing coefficients; it did not fit them.
Store also used validation for permitted discrete selection. Neither result is
an unseen-test claim. Overall MAPE is the acceptance criterion; per-group,
direction and repeat quality are reports. Two Store repeat failures remain;
follow-up collection was cancelled. Nonpositive labels are not filtered by APE.

Current Load arithmetic checks cover 7088 inputs and 843472 features (zero
feature difference, maximum prediction difference 1.46e-11), 496 Device poison
pointers and 7088 missing-type rejections. IR-adapter checks cover 104 positive
routes, 416 refusals, 104 compositions, 16 Host/Device boundaries, four FP8 cases
and 352 mask patterns. These are CPU compiler checks, not hardware measurements.
Historical Store arithmetic covers 15128 inputs; historical extension builds
and removed Python-API tests do not certify the current scoped revision.
Complete current-extension and owned/event/F>1 hardware acceptance are not
claimed. CI must be checked on the actual commit.
