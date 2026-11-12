# Release image inputs

`cmake/release-image-info.json` is the source of truth for the selected release
image: wheel version, immutable source commit, LLVM revision, patch digests,
CANN, PyTorch, torch-npu and Python versions. This is a frozen release-scoped
snapshot, independent of main's current compiler and development-image defaults.

The initial snapshot uses the v3.6.0 source commit
`97e8a2be1e46817c359f55c0e6464e49480d0b1f`. Its LLVM revision and patch digest
were extracted from that commit. Updating main's LLVM must not rewrite this
release snapshot. Changing the selected release requires reviewing all fields
together. Source metadata does not authenticate the downloaded wheel binary.

The release workflow validates and resolves these inputs once with jq for the
build and manifest jobs. The Dockerfile uses its already installed Python to
check arguments and verify downloaded patches, without another parser dependency.
It requires those resolved version arguments and rejects a
partial override that differs from the snapshot. It downloads LLVM patches from
the immutable release source, verifies their SHA-256 and applies them before
building LLVM. It has no independent LLVM_COMMIT argument.

For a local build, pass the same arguments as the release workflow's build-args
block, resolving them from this JSON. Infrastructure choices such as base OS,
device type, architecture and package mirror remain separate build arguments.

Historical image recipes retain their original release inputs. Runtime
compatibility certification and complete binary provenance remain separate
requirements; neither a metadata test nor Docker's check mode is a full build.
