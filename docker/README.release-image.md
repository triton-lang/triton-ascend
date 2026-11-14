# Release image inputs

`cmake/release-image-info.json` is the source of truth for the selected release
image: wheel version, immutable source commit, LLVM revision, patch digests,
CANN (Compute Architecture for Neural Networks), PyTorch, torch_npu and
Python versions. This is a frozen release-scoped
snapshot, independent of main's current compiler and development-image defaults.

The initial snapshot uses the v3.6.0 source commit
`97e8a2be1e46817c359f55c0e6464e49480d0b1f`. Its LLVM revision and patch digest
were extracted from that commit. Updating main's LLVM must not rewrite this
release snapshot. Changing the selected release requires reviewing all fields
together. The Triton wheel built inside the image from this pinned source is
what the runtime consumes; the metadata snapshot itself does not authenticate
the resulting wheel binary.

The release workflow validates and resolves these inputs once with jq for the
build and manifest jobs. The Dockerfile uses its already installed Python to
check arguments and verify downloaded patches, without another parser dependency.
It requires those resolved version arguments and rejects a
partial override that differs from the snapshot. It downloads LLVM patches from
the immutable release source, verifies their SHA-256 and applies them before
building LLVM. It has no independent LLVM_COMMIT argument.

For a local build, pass the same version arguments as the release workflow's
[build-args block](../.github/workflows/release-docker-images-nightly.yml)
(`docker/triton_ascend-cann-torch_npu.Dockerfile` is the build file), resolving
them from this JSON. BuildKit is required (the recipe relies on
`TARGETARCH`); run the command from the repository root so the Dockerfile
path and the in-image metadata file resolve. Infrastructure arguments
(`BASE_IMAGE`, `DEVICE_TYPE`, `PIP_SOURCE`) are omitted below and keep their
Dockerfile defaults. For the current snapshot, with the values resolved from
`cmake/release-image-info.json`:

```bash
docker build \
    --build-arg TRITON_ASCEND_VERSION=3.6.0 \
    --build-arg TRITON_ASCEND_SOURCE_SHA=97e8a2be1e46817c359f55c0e6464e49480d0b1f \
    --build-arg CANN_VERSION=9.2.0 \
    --build-arg PYTORCH_VERSION=2.10.0 \
    --build-arg TORCH_NPU_VERSION=2.10.0.post7 \
    --build-arg PYTHON_VERSION=3.11.15 \
    --build-arg PYTHON_SERIES=3.11 \
    -f docker/triton_ascend-cann-torch_npu.Dockerfile \
    -t triton-ascend:3.6.0-cann9.2.0-local .
```

| Build argument | JSON field |
| --- | --- |
| `TRITON_ASCEND_VERSION` | `triton_version` |
| `TRITON_ASCEND_SOURCE_SHA` | `source_hash` |
| `CANN_VERSION` | `cann` |
| `PYTORCH_VERSION` | `torch` |
| `TORCH_NPU_VERSION` | `torch_npu` |
| `PYTHON_VERSION` | `python_version` |
| `PYTHON_SERIES` | `python_series` (derived; see the note below the table) |

`PYTHON_SERIES` is a derived field, not present in
`release-image-info.json`: the release workflow computes it from the first
two components of `python_version` (e.g. `3.11.15` → `3.11`) and still
passes it explicitly as a build argument; the image checks that it matches
the derived value.

Beyond the arguments in the table, the `llvm_hash` and `patches` fields are
consumed inside the image from the metadata file itself and are not build
arguments.

Infrastructure choices such as base OS, device type and package mirror remain
separate build arguments (`BASE_IMAGE`, `DEVICE_TYPE`, `PIP_SOURCE`); the
target architecture is supplied automatically by BuildKit's `TARGETARCH`
build argument and is not passed manually.

Historical image recipes retain their original release inputs. Runtime
compatibility certification and complete binary provenance remain separate
requirements; neither a metadata test nor Docker's check mode is a full build.
