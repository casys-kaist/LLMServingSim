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
current table contract, required measurement inputs, CPU-only rebuild command
dynamic acquisition, resumable repetitions, independent skew-only TP selection,
automatic reference-cell support completion, and migration of enabled bundles.
The [hardware guide](docs/profiler/adding-hardware.md) documents standalone
hardware characterization, backend-aligned Ring calibration, bandwidth units,
recorded residuals and the distinction from grouped MoE communication.
The [bench reference](docs/reference/bench-cli.md#parallelism) documents the
NCCL baseline and recorded communicator and compilation settings.
The [parallelism guide](docs/simulator/parallelism-mechanics.md) distinguishes
local CUDA graph padding from DP synchronization; target graph overrides live
in the [cluster reference](docs/reference/cluster-config.md#cuda-graph-contract).
The [expert routing guide](docs/simulator/moe-expert-routing.md) distinguishes
global EP-rank lookup from instance-local trace markers.

## Installation

```bash
yarn
```

## Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
