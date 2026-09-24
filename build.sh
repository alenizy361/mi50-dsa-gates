#!/usr/bin/env bash
# Builds the four HIP gate binaries for gfx906 into ./bin
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin
HIPCC=${HIPCC:-hipcc}
ARCH=${ARCH:-gfx906}
ROCM=${ROCM_PATH:-/opt/rocm}
FLAGS="--offload-arch=${ARCH} -O3 -std=c++17 -Wall -Igates"
LIBS="-L${ROCM}/lib -lhsa-runtime64"   # after the source: GNU ld resolves libraries left to right
HIPCC_VER=$($HIPCC --version 2>/dev/null || true)
if [ -n "$HIPCC_VER" ]; then echo "hipcc: ${HIPCC_VER%%$'\n'*}"; else echo "hipcc: not found (is $HIPCC on PATH?)"; exit 1; fi
$HIPCC $FLAGS -o bin/p2p_matrix        gates/01_p2p_matrix.hip        $LIBS
$HIPCC $FLAGS -o bin/peer_flag         gates/02_peer_flag.hip         $LIBS
$HIPCC $FLAGS -o bin/oneshot_allreduce gates/03_oneshot_allreduce.hip $LIBS
$HIPCC $FLAGS -o bin/gemv_q8_bw        gates/04_gemv_q8_bw.hip        $LIBS
echo "built: bin/p2p_matrix bin/peer_flag bin/oneshot_allreduce bin/gemv_q8_bw"
