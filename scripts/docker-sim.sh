#!/bin/bash

# Launch the simulator Docker container (ASTRA-Sim + sim Python deps).
#
# Mounts the repo root regardless of where this script is invoked from.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                    # .../LLMServingSim

# The image ships Python 3.10 and the ASTRA-Sim build deps, but none of the
# packages the Python side needs. Every one below is imported by code that runs
# in this container:
#   pyyaml        profiler meta.yaml and the architecture catalogs
#   pyinstrument  llmservingsim/serving/__main__.py
#   rich          llmservingsim/serving/ and bench/ loggers
#   pandas        scheduler, trace_generator, pim_model
#   numpy         scheduler
#   matplotlib    llmservingsim/bench/core/plots.py, reached by `python -m llmservingsim.bench validate`
# The versions are pinned for this image, so the repo goes in with --no-deps:
# that is only for the import paths, so `import llmservingsim` works from any cwd.
docker run --name servingsim_docker \
  -it \
  -v "$REPO_ROOT":/app/LLMServingSim \
  -w /app/LLMServingSim \
  astrasim/tutorial-micro2024 \
  bash -c "pip3 install pyyaml pyinstrument rich \
  pandas==1.5.3 numpy==1.23.5 matplotlib==3.5.3 && \
  pip3 install -e . --no-deps && exec bash"