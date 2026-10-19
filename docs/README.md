# docs

Public documentation website built with [Docusaurus](https://docusaurus.io/).

## Layout

Selected source directories and build entry points:

```text
docs/
├── docs/                   user and contributor Markdown/MDX guides
├── src/                    site pages, components and styles
│   └── pages/changelog.md  generated from the root CHANGELOG.md
├── static/                 public images and deployment assets
├── scripts/                changelog synchronization and render checks
├── docusaurus.config.ts    site configuration
├── sidebars.ts             explicit guide navigation
└── package.json            development, build and validation commands
```

## Contribution policy

Follow the repository [commit policy](docs/contributor/pr-workflow.md#commit-hygiene)
and [AGENTS.md](../AGENTS.md). Every commit includes updates to the root and
affected directory READMEs, AGENTS.md, CHANGELOG.md, and relevant public docs.
Publish supported behavior and usage, not session notes or intermediate results;
temporary diagnostic and verification scripts or tests remain uncommitted.
Regenerate the site changelog from the root CHANGELOG.md and run a production
build before handing off documentation changes.

For release checks, compare every public parser's options with its README
index and full CLI reference, including wrapper-only defaults, boolean
negations and subcommand restrictions. The profiler running guide also
lists all resource-watchdog options and distinguishes unset from resolved
engine settings.

The changelog is a release summary, not a commit log. Use one heading per change
category within each release, consolidate related pending entries, and keep
upgrade requirements explicit. See the
[changelog entry guidelines](docs/contributor/pr-workflow.md#changelog-entries).

Accuracy claims refer to the recorded bundles and workloads on the
[validation page](docs/validation.md), not every architecture catalog or newly
acquired profile. Keep GPU recording separate from CPU-only comparison in setup
instructions. Root Getting Started keeps runnable commands, runtime requirements
and documentation links; measurement details, model-specific limits and
contribution policies stay in the topic guides. Follow the shared
[README formatting conventions](docs/contributor/conventions.md#readme-and-markdown-formatting)
for directory trees and examples. Directory introductions stay short; detailed
contracts belong under topic headings after the usage entry points.
Root Latest News entries link shipped changes
to their PR and label future backend/model work as in progress.

## Guide responsibilities

Keep detailed contracts in their topic guides instead of duplicating them in
this index. During a release pass, check copy-pasted commands against the actual
wrappers: an editable template can overwrite an exported environment variable.
Distinguish accepted arguments from applied behavior and explicit refusals.

| Topic | Authoritative guide |
| --- | --- |
| GPU setup, package pins and retained source patches | [vLLM installation](docs/getting-started/installation/vllm.mdx) |
| CLI options, sampling grids and resume/force conditions | [Running the profiler](docs/profiler/running.md) |
| CUDA-only timing, dummy preparation, root metadata and native support limits | [Profile bundle](docs/profiler/output-bundle.md) |
| Local and DP+EP MoE acquisition, routing axes and deployment coverage | [Native MoE components](docs/profiler/native-moe-components.md) |
| Raw skew measurements and explicit CPU-only calibration | [Skew calibration](docs/profiler/skew-alpha-fit.md) |
| Collective measurement, units and configuration precedence | [Adding hardware](docs/profiler/adding-hardware.md) and [cluster configuration](docs/reference/cluster-config.md) |
| Recorded backend, per-engine KV capacity and serving graph observations | [Bench CLI](docs/reference/bench-cli.md) |
| Flat benchmark inputs, workload generation and agentic simulation boundaries | [ShareGPT generator](docs/workloads/sharegpt-generators.md) and [agentic sessions](docs/workloads/agentic-sessions.md) |
| Kimi/native DeepSeek duration replay versus ordinary serving | [Trace generation](docs/simulator/trace-generation.md) |
| Logical bytes, native MoE payloads and unsupported traffic/capacity claims | [Trace format](docs/reference/trace-format.md) |
| Official reference scopes and all latency statistics | [Validation](docs/validation.md) |
| Submodule publication and repeatable regression checks | [PR workflow](docs/contributor/pr-workflow.md) and [validation workflow](docs/contributor/validating-changes.md) |

The RTXPRO6000 headline references cover Llama TP1, Qwen3-32B TP2 and
Qwen3-30B TP1/DP2/EP2 with refreshed vLLM 0.31.0 data. Acquiring another TP
or adding a catalog does not extend those accuracy claims. Kimi's local
duration replay and native DeepSeek's duration-only representation remain
distinct from ordinary scheduler/capacity admission; the guides state the limits.

## Installation

Use Node.js 20 or newer and pnpm. The deployment workflow pins Node.js 22.

```bash
pnpm install --frozen-lockfile
```

## Local development

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
