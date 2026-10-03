#!/usr/bin/env bash
# Linux installation wrapper. No checkpoints are downloaded by this script.
set -e

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo 'Usage: bash setup.sh [--new-env]'
    echo 'Requires Linux, git, conda and CUDA development tools. Delegates to the official installer.'
    exit 0
fi

TASK_WRAPPER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_TRELLIS_ROOT="${TRELLIS2_ROOT:-$TASK_WRAPPER_ROOT/TRELLIS.2}"
TASK_TRELLIS_REF="${TRELLIS2_REF:-75fbf0183001ed9876c8dbb35de6b68552ee08bd}"

command -v git >/dev/null || { echo 'git is required'; exit 1; }
command -v conda >/dev/null || { echo 'conda is required; see README'; exit 1; }
eval "$(conda shell.bash hook)"

if [[ ! -d "$TASK_TRELLIS_ROOT" ]]; then
    git clone --recursive https://github.com/microsoft/TRELLIS.2.git "$TASK_TRELLIS_ROOT"
    git -C "$TASK_TRELLIS_ROOT" checkout "$TASK_TRELLIS_REF"
    git -C "$TASK_TRELLIS_ROOT" submodule update --init --recursive
else
    TASK_EXISTING_REF="$(git -C "$TASK_TRELLIS_ROOT" rev-parse HEAD)"
    if [[ "$TASK_EXISTING_REF" != "$TASK_TRELLIS_REF" ]]; then
        echo "Existing checkout is $TASK_EXISTING_REF; expected $TASK_TRELLIS_REF."
        echo 'Inspect your checkout and explicitly set TRELLIS2_REF to its full commit to keep that version.'
        exit 1
    fi
fi

cd "$TASK_TRELLIS_ROOT"
# The upstream installer is sourced so its conda activation applies to the extra dependencies.
. ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm "$@"
python -m pip install -r "$TASK_WRAPPER_ROOT/requirements.txt"
echo 'Installation commands completed. Activate trellis2 if --new-env was used.'
echo 'Download checkpoints separately: python download_weights.py'
