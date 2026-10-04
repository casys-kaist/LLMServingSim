# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Contribution policy

Follow the repository [commit policy](docs/contributor/pr-workflow.md#commit-hygiene)
and [AGENTS.md](../AGENTS.md). Every commit includes updates to the root and
affected directory READMEs, AGENTS.md, CHANGELOG.md, and relevant public docs.
Publish supported behavior and usage, not session notes or intermediate results;
temporary diagnostic and verification scripts or tests remain uncommitted.
Regenerate the site changelog from the root CHANGELOG.md and run a production
build before handing off documentation changes.

The [skew calibration guide](docs/profiler/skew-alpha-fit.md) documents the
current table contract, required measurement inputs, CPU-only rebuild command,
dynamic acquisition, resumable repetitions, independent skew-only TP selection,
automatic reference-cell support completion, and migration of enabled bundles.
It also describes common-metadata geometry validation for sparse indexers
whose backend-specific metadata does not retain query boundaries.
Skew timing documentation separates launch-correlated kernel ownership from
latency: CPU scopes identify the launching module but contribute no time.
The [output guide](docs/profiler/output-bundle.md#cuda-activity-and-acquisition-identity)
defines per-call CUDA interval unions, acquisition identity columns and
incompatible-resume protection; existing tables are not automatically converted.
Its whole-block MoE section documents the shared fine token grid and the
distinction between acquisition resolution and runtime interpolation.
The resume guide also distinguishes atomic progress checkpoints from complete
bundles ready for simulation and documents exact fractional attention-key matching.
The acquisition guides document assigned-page dummy KV initialization outside
timed query work, its content/layout limitations and asynchronous completion.
They also distinguish request phase from query/history geometry: native decode
requests need a completed prompt and separate output tokens for correct ordering.
The [hardware guide](docs/profiler/adding-hardware.md) documents standalone
hardware characterization, backend-aligned Ring calibration, bandwidth units,
recorded residuals and the distinction from grouped MoE communication.
The [collective-link schema](docs/reference/cluster-config.md#collective-specific-links)
documents optional per-operation analytical Ring links and configuration
precedence, without changing collective payloads or the common fallback.
The [native MoE guide](docs/profiler/native-moe-components.md) describes explicit-DP
single-rank acquisition, publication checks, component lookup and supported
execution contracts. It distinguishes measured GPU work from analytical transport.
The [profile table contract](docs/profiler/output-bundle.md#moecsv-legacy-whole-block-moe-profiles)
also explains whole-block MoE invocation normalization and when existing
hybrid-stack tables need remeasurement. Forced-routing grids retain native
top-k GPU work, warm the requested distribution, and reject bypassed hooks;
older grids that omitted routing kernels also require a refresh.
The same table guide explains non-overlapping sparse-indexer glue ownership
and the dense-category refresh needed for older DeepSeek/GLM glue rows.
The [validation page](docs/validation.md) reports the bundled NCCL-only Qwen3-32B
and native Qwen3-30B examples, both inheriting the same hardware operation defaults,
and keeps each recorded reference matched to its transport calibration.
The [bench reference](docs/reference/bench-cli.md#parallelism) documents the
NCCL baseline and recorded communicator and compilation settings.
It also documents phase-delimited gate observations and their exclusion from
unperturbed latency references, without discarding concentrated workload gates.
The [parallelism guide](docs/simulator/parallelism-mechanics.md) distinguishes
local CUDA graph padding from DP synchronization; target graph overrides live
in the [cluster reference](docs/reference/cluster-config.md#cuda-graph-contract).
It also separates non-speculative head rows from forward padding and explains
why idle DP forwards run the backbone without logits or sampling, with
independent TP/EP collective numbering in the backend.
The [expert routing guide](docs/simulator/moe-expert-routing.md) distinguishes
global EP-rank lookup from instance-local trace markers, and documents
round-robin top-k assignment over gathered token positions.
The [publication workflow](docs/contributor/pr-workflow.md#publishing-submodule-changes)
explains how to publish Chakra, ASTRA-Sim and the frontend without leaving
unavailable submodule commits in a public checkout.

## Installation

Use Node.js 20 or newer and pnpm. The deployment workflow pins Node.js 22.

```bash
pnpm install --frozen-lockfile
```

## Local Development

```bash
pnpm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
pnpm build
pnpm check-rendered
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

The GitHub Actions workflow in `.github/workflows/deploy-docs.yml` builds and
deploys the site on documentation changes pushed to `main`. A feature-branch
push does not deploy the public site. Do not publish a separate `gh-pages`
branch with the generic Docusaurus deploy command.
