# Contributors

LLMServingSim is developed and maintained by the
[CASYS](https://casys.kaist.ac.kr) research group at KAIST. It would not be
what it is without the people who have given their time, insight, and code to
the project. This page is our way of saying **thank you**.

## Core Team — CASYS, KAIST

- **Jaehong Cho** ([@JaehongCS20](https://github.com/JaehongCS20))
- **Hyunmin Choi** ([@hyuenmin-choi](https://github.com/hyuenmin-choi))
- **Guseul Heo**
- **Minsu Kim**
- **Jongse Park** — faculty advisor

## Community Contributors

We are especially grateful to contributors from outside CASYS who have
volunteered their effort to make LLMServingSim better for everyone. 🙏

### Code

- **[@horser1](https://github.com/horser1)**
  - Per-dimension link settings + collective dimension sync ([#33](https://github.com/casys-kaist/LLMServingSim/pull/33))
  - Prefix-cache / radix-tree fixes ([#35](https://github.com/casys-kaist/LLMServingSim/pull/35))
  - Per-instance runtime config overrides ([#37](https://github.com/casys-kaist/LLMServingSim/pull/37))
  - Non-DP multi-instance collective scoping ([#39](https://github.com/casys-kaist/LLMServingSim/pull/39))
  - Run-isolated ASTRA-Sim input paths ([#43](https://github.com/casys-kaist/LLMServingSim/pull/43))
  - KV eviction/reload accounting ([#48](https://github.com/casys-kaist/LLMServingSim/pull/48))
  - Clean up intermediate ASTRA-Sim inputs to save storage ([#51](https://github.com/casys-kaist/LLMServingSim/pull/51))
  - Fix argument shadowing in `serving/__main__.py` ([#53](https://github.com/casys-kaist/LLMServingSim/pull/53))
- **[@Veilwalker](https://github.com/Veilwalker)**
  - Avoid duplicate prefix-cache hit accounting under chunked prefill ([#49](https://github.com/casys-kaist/LLMServingSim/pull/49))
- **[@zsxh1990](https://github.com/zsxh1990)**
  - Docs for per-instance runtime overrides ([#38](https://github.com/casys-kaist/LLMServingSim/pull/38))
  - Generalized PIM latency model for arbitrary architectures ([#45](https://github.com/casys-kaist/LLMServingSim/pull/45))
- **[@shermanjlim](https://github.com/shermanjlim)**
  - `avail_size()` overestimation and `storage_cache_evicted_req` fixes ([#29](https://github.com/casys-kaist/LLMServingSim/pull/29))
- **[@Snowfall99](https://github.com/Snowfall99)**
  - Fix simulator docs pagination ([#56](https://github.com/casys-kaist/LLMServingSim/pull/56))
- **[@hsule](https://github.com/hsule)**
  - Fix stale `--dtype float16` in `run.sh` and the docs examples ([#57](https://github.com/casys-kaist/LLMServingSim/pull/57))
- **[@gleb-kun](https://github.com/gleb-kun)**
  - Fix missing return value in the profiler's argument parser ([#22](https://github.com/casys-kaist/LLMServingSim/pull/22))
- **[@Arifuzzamanjoy](https://github.com/Arifuzzamanjoy)**
  - RTX 4090 profile bundle with an end-to-end vLLM validation run, which exposed
    the attention-interpolation and skew-fallback errors
    ([#58](https://github.com/casys-kaist/LLMServingSim/issues/58),
    [#59](https://github.com/casys-kaist/LLMServingSim/pull/59))

### Reports and analysis

Issues that pinned down a real problem, and the digging that made the fix
possible. These shaped the project as much as the patches did.

- **[@hu-op1](https://github.com/hu-op1)**
  - Accuracy boundary report against vLLM V1, which set the direction the KV cache
    and scheduler were rebuilt in ([#40](https://github.com/casys-kaist/LLMServingSim/issues/40))
  - Pipeline parallelism silently hung at most `pp_size` values ([#55](https://github.com/casys-kaist/LLMServingSim/issues/55))
- **[@hsule](https://github.com/hsule)**
  - Narrowed the pipeline-parallelism hang to the stage-boundary send/recv size
    mismatch ([#55](https://github.com/casys-kaist/LLMServingSim/issues/55))
  - `pp > 1` could complete the same request twice, corrupting the per-request
    CSV and raising `KeyError` in `cache_blocks` with prefix caching on
    ([#62](https://github.com/casys-kaist/LLMServingSim/issues/62))
  - DP groups hung with no error whenever combined with `tp > 1` or `pp > 1`
    ([#65](https://github.com/casys-kaist/LLMServingSim/issues/65))
- **[@bui-thanh-lam](https://github.com/bui-thanh-lam)**
  - Model-architecture YAML documentation had drifted from the schema
    ([#52](https://github.com/casys-kaist/LLMServingSim/issues/52))

If you have contributed and are not listed here, or you'd like your entry
updated, please open a pull request or
[reach out](https://llmservingsim.ai/contact) — we want everyone's work to be
recognized.

## Acknowledgments

The base layerwise-profile methodology in `profiler/` is adapted from
[@waneon](https://github.com/waneon). LLMServingSim builds on
[ASTRA-Sim](https://github.com/astra-sim/astra-sim) and
[Chakra](https://github.com/mlcommons/chakra).

The KV cache is a port of [vLLM](https://github.com/vllm-project/vllm)'s block
pool — per-tier free-block queues, chained block hashes, eviction as a side
effect of allocation — and the scheduler follows vLLM V1's two-phase
`schedule()`. Before that, prefix caching was built on a radix tree adapted from
[SGLang](https://github.com/sgl-project/sglang), which served the project through
several releases.

---

Interested in contributing? See the
[contributor guide](https://llmservingsim.ai/docs/contributor/welcome).
