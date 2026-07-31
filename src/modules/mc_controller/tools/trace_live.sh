#!/bin/bash
# Snapshot the live control chain while the vehicle is running.
#
# The post-flight tracer (trace_chain.py) reads a ULog, so it can only ever see
# logger-downsampled rates - actuator_motors is capped at 100 ms and looks like
# 10 Hz when it is really running at gyro rate. This script asks the running
# system instead, which is the only source of TRUE publication rates.
#
#   ./trace_live.sh                      # one snapshot to stdout
#   ./trace_live.sh -o trace.txt         # to a file
#   ./trace_live.sh -n 5 -i 10           # 5 snapshots, 10 s apart (watch it evolve)
#
# Run it while armed and flying - rates for a disarmed vehicle are meaningless.

set -o pipefail
PX4=${PX4_BIN:-build/px4_sitl_default/bin}
OUT=/dev/stdout
N=1
INTERVAL=5

while getopts "o:n:i:h" opt; do
  case $opt in
    o) OUT=$OPTARG ;;
    n) N=$OPTARG ;;
    i) INTERVAL=$OPTARG ;;
    h) sed -n '2,14p' "$0"; exit 0 ;;
    *) exit 1 ;;
  esac
done

if [ ! -x "$PX4/px4-uorb" ] && [ ! -x "$PX4/px4-work_queue" ]; then
  echo "px4 shell wrappers not found in $PX4" >&2
  echo "set PX4_BIN, or on hardware run these commands over the MAVLink console" >&2
  exit 1
fi

run() { "$PX4/px4-$1" "${@:2}" 2>&1; }

{
for i in $(seq 1 "$N"); do
  echo "================================================================================"
  echo "LIVE CHAIN SNAPSHOT $i/$N   $(date -u +%H:%M:%SZ)"
  echo "================================================================================"

  echo
  echo "--- which controller is actually active (never assume from a param) ---"
  run mc_controller status | grep -aE "controller|level|fallback|invalid|outer|queues|updates" \
    || echo "    mc_controller not running -> stock cascade"

  echo
  echo "--- work queue placement: what runs where, and how often ---"
  echo "    (interval_us is the scheduling period; rate = 1e6/interval_us)"
  run work_queue status | grep -aE "rate_ctrl|nav_and_controllers|INS|mc_|control_alloc|ekf2" \
    || echo "    unavailable"

  echo
  echo "--- TRUE publication rates (this is what the ULog cannot tell you) ---"
  run uorb top -1 | grep -aE \
    "TOPIC|angular_velocity|vehicle_attitude|local_position|torque_setpoint|thrust_setpoint|rates_setpoint|actuator_motors|trajectory_setpoint|control_mode|attitude_setpoint|mc_controller_status|hover_thrust" \
    || echo "    unavailable"

  echo
  echo "--- per-module cycle time (HARDWARE ONLY: reads 0us in SITL, stock included) ---"
  run perf | grep -aiE "mc_controller|mc_rate_control|mc_att_control|mc_pos_control|control_alloc" \
    || echo "    unavailable"

  echo
  echo "--- publisher/subscriber counts: >1 publisher on an output topic is a bug ---"
  for t in vehicle_torque_setpoint vehicle_thrust_setpoint vehicle_attitude_setpoint \
           vehicle_rates_setpoint vehicle_local_position_setpoint actuator_motors; do
    line=$(run uorb status | grep -a " $t ")
    [ -n "$line" ] && echo "    $line"
  done

  [ "$i" -lt "$N" ] && sleep "$INTERVAL"
done

echo
echo "================================================================================"
echo "Reading this:"
echo "  - Two publishers on one output topic means contention. The framework and"
echo "    two work items both publishing a setpoint is a defect (it has happened here)."
echo "  - The inner loop must sit at IMU_GYRO_RATEMAX (400 on hardware, 250 in SITL)."
echo "    Lower means the gyro callback is being starved."
echo "  - perf cycle time is the only real measure of whether your law fits the"
echo "    budget: 2.5 ms at 400 Hz. It is unavailable in SITL."
echo "  - Pair this with the post-flight view:  trace_chain.py <flight.ulg> --values"
echo "================================================================================"
} > "$OUT"
