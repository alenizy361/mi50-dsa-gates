#!/usr/bin/env bash
# Gate 5 — all-GPU soak: does the HBM clock, temperature or bandwidth sag under a sustained
# memory-bound loop (the decode regime)? Runs gemv_q8_bw on every GPU for DUR seconds and logs
# rocm-smi telemetry every 5 s — only while every GPU is actually under load. Needs no P2P.
#   gates/05_thermal_soak.sh [seconds=600]
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PATH:${ROCM_PATH:-/opt/rocm}/bin"
DUR=${1:-600}
mkdir -p results
if [ -n "${HIP_VISIBLE_DEVICES:-}${ROCR_VISIBLE_DEVICES:-}" ]; then
  echo "unset HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES: this script assigns one GPU per process itself"; exit 1
fi
N=$(./bin/gemv_q8_bw --count 2>/dev/null) || N=
if [ -z "$N" ] || ! [ "$N" -ge 1 ] 2>/dev/null; then echo "no HIP devices, or bin/gemv_q8_bw missing — run ./build.sh first"; exit 1; fi
command -v rocm-smi >/dev/null || { echo "rocm-smi not on PATH"; exit 1; }
TS=$(date +%Y%m%d_%H%M%S)
CSV=results/gate5_soak_${TS}.csv
LOGP=results/gate5_soak_${TS}
echo "soaking $N GPU(s) for ${DUR}s; telemetry -> $CSV"
{ echo "### bus map"; rocm-smi --showbus --csv; echo "### power cap"; rocm-smi --showmaxpower --csv; } >> "$CSV" 2>/dev/null
pids=()
for g in $(seq 0 $((N - 1))); do
  HIP_VISIBLE_DEVICES=$g ./bin/gemv_q8_bw --shape 6144x6144 --seconds "$DUR" > "${LOGP}_gpu$g.log" 2>&1 &
  pids+=($!)
done
# Start sampling only once EVERY running GPU has printed its first 'soak t=' line (~10 s into its load
# loop): host fill, H2D and warm-up are over, so no idle-clock sample enters the CSV.
t0=$SECONDS
ready=0; alive=0
while :; do
  ready=0; alive=0
  for g in $(seq 0 $((N - 1))); do grep -q '^soak t=' "${LOGP}_gpu$g.log" 2>/dev/null && ready=$((ready + 1)); done
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
  [ "$ready" -eq "$N" ] && break
  if [ "$alive" -eq 0 ]; then echo "ERROR: no GPU reached the soak loop; see ${LOGP}_gpu*.log" >&2; wait; exit 1; fi
  if [ $((ready + N - alive)) -ge "$N" ] || [ $((SECONDS - t0)) -ge 120 ]; then
    echo "WARN: only $ready/$N GPU(s) reached the soak loop ($alive still running); see ${LOGP}_gpu*.log" >&2; break
  fi
  sleep 1
done
[ "$ready" -ge 1 ] || { wait; exit 1; }
for g in $(seq 0 $((N - 1))); do echo "### hip gpu$g $(grep -m1 -oE 'pci=[0-9a-f:.]+' "${LOGP}_gpu$g.log")"; done >> "$CSV"   # HIP index -> PCI
# The load ends ~DUR-10 s from here; stop early so no post-load (idle) sample enters either.
end=$((SECONDS + (DUR > 60 ? DUR - 25 : DUR / 2)))
while [ $SECONDS -lt $end ]; do
  echo "### $(date +%s)" >> "$CSV"
  rocm-smi --showtemp --showclocks --showpower --csv >> "$CSV" 2>/dev/null
  sleep 5
done
wait
echo "== per-GPU GB/s: first sample -> last sample (sag > 5% = throttling)"
for g in $(seq 0 $((N - 1))); do
  first=$(grep '^soak' "${LOGP}_gpu$g.log" | head -1 | grep -oE 'GB/s=[0-9.]+')
  last=$(grep '^soak' "${LOGP}_gpu$g.log" | tail -1 | grep -oE 'GB/s=[0-9.]+')
  echo "gpu$g: ${first:-n/a} -> ${last:-n/a}"
done
python3 - "$CSV" <<'EOF'
import sys, csv, re, collections, os
if not os.path.exists(sys.argv[1]):
    print("no telemetry captured (DUR too short or rocm-smi missing)"); sys.exit(0)
rows = [r for r in csv.reader(open(sys.argv[1])) if r]
hdr = None; in_samples = False; cap = {}
data = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if r[0].startswith('#'):
        in_samples = in_samples or r[0][3:].strip().isdigit()   # '### <epoch>' opens the load samples
        continue
    if r[0].strip().lower() == 'device':
        hdr = r; continue
    if not hdr or len(r) != len(hdr) or not r[0].startswith('card'):
        continue
    for k, v in zip(hdr[1:], r[1:]):
        try:
            val = float(re.sub(r'[^0-9.]', '', v))
        except ValueError:
            continue
        if in_samples: data[r[0]][k].append(val)
        elif 'power' in k.lower(): cap[r[0]] = val
if not data:
    print('no load samples in the CSV (soak never became ready, or rocm-smi failed)')
for dev in sorted(data):
    d = data[dev]; out = [dev]
    for k in d:
        kl = k.lower()
        if not d[k]: continue
        if 'temperature' in kl: out.append(f"{k}: max {max(d[k]):.0f}")
        elif 'clock speed' in kl and ('mclk' in kl or 'sclk' in kl): out.append(f"{k}: min {min(d[k]):.0f} max {max(d[k]):.0f}")
        elif 'power' in kl: out.append(f"{k}: max {max(d[k]):.0f}")
    if dev in cap: out.append(f"power cap {cap[dev]:.0f} W")
    print(' | '.join(out))
print("PASS if: mclk min == mclk max during load (flat), memory-sensor temperature stays under the throttle point, and last-sample GB/s >= 95% of the first.")
print("Join HIP gpuN (GB/s lines) to rocm-smi cardN through the '### hip' (pci=) and '### bus map' rows at the top of the CSV.")
EOF
echo "saved: $CSV ${LOGP}_gpu*.log"
