# Optional Ascend source dependencies

`triton-distributed-info.json` owns the optional distributed backend repository
and exact source revision. Preparation reads this metadata and verifies the
checked-out commit, including outside a Git source checkout.

`npuir-info.json` owns the compiler repository and watch branch. Both source
fetching and wheel-cache resolution use the same script. Its `--resolve-only`
mode does not modify the destination. Explicit repository/branch overrides
remain available, and metadata parsing requires Python 3. The wheel job prepares
that interpreter before invoking the script.

Wheel build, test and publication share the selected push/manual source ref and
artifact tag. The Ubuntu ref-selection job uses Bash and jq, without preparing
an extra Python interpreter. Pull requests explicitly select the head commit. Post-submit
throttling queries the current branch and CI workflow filename.

These settings do not define a complete runtime compatibility matrix or certify
compiler ABI compatibility. A moving watch branch is not a frozen release lock;
source identity and artifact provenance have separate validation requirements.
