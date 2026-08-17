#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# Intall Chakra (Use Chakra fork in ASTRA-Sim repo).
(
cd ${REPO_ROOT}/astra-sim/extern/graph_frontend/chakra
pip3 install .
)

# Compile ASTRA-sim with analytical backend model
(
cd ${REPO_ROOT}/astra-sim
bash ./build/astra_analytical/build.sh
)

# Compile ASTRA-sim with ns3 backend model
# (
# cd ${REPO_ROOT}/astra-sim
# bash ./build/astra_ns3/build.sh
# )
