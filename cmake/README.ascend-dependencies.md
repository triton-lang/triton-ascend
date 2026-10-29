# Source build dependency inputs

Follow the existing Triton/Intel layout: LLVM metadata is JSON and the
nlohmann-json tag remains in `json-version.txt`. Each migrated source dependency
has one authoritative pin; a single file format is not required.

| File | Consumers |
| --- | --- |
| `llvm-info.json` | Full LLVM SHA and sorted Ascend patch manifest. Packaging, the LLVM source-build workflow, wrapper and four installation sources read their own checkout. |
| `json-version.txt` | JSON-library download URL in `python/build_helpers.py`, using `read().strip()` as upstream does. |
| `bisheng-info.json` | Installer names and SHA-256 for both architectures; four CI installation steps read and verify these inputs. |

Common package and platform helpers live in the existing
`python/build_helpers.py`. Ascend-specific URLs, patch-content hashes and the
AlmaLinux ABI exception stay in `third_party/ascend/build/setup_patch.py`.
Explicit toolchain paths and offline behavior retain their existing roles.
CMake continues to discover installed LLVM/MLIR configs and JSON includes.
No generic reader, generated mirror or metadata-only CMake module is required.
Workflow shell steps use `jq -er` for metadata fields. The LLVM container
installs jq before reading its pin. Python packaging and developer commands
use the Python standard library, without adding a jq dependency there.

`scripts/check_bom_consistency.py` audits these declared inputs, local patch
names/contents, the four installation sources, the main release-policy LLVM/patch
columns, and workflow cache paths. It does not supply production values or
certify arbitrary binaries, runtime environments or complete release matrices.

To update a source dependency, edit its authoritative file and applicable local
patches together. Run `python scripts/check_bom_consistency.py` and
`python -m pytest -q third_party/ascend/build_tests`.
Compiler cache keys include the actual metadata and patch bytes; unrelated
documentation changes do not rotate that portion of the key.

Runtime compatibility groups, optional dependency source policies, release
images and download cleanup are separate changes described in RFC 2739.
This change leaves runtime image/torch defaults and historical release recipes
in their existing roles. Source distribution inclusion is provided by the
existing manifest; it does not replace a real build/install check.
