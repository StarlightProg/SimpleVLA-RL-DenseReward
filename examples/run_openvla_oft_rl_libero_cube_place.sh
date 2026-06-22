#!/usr/bin/env bash
set -euo pipefail

export REWARD_MODE=sparse
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_openvla_oft_rl_libero_cube_place_common.sh" "$@"
