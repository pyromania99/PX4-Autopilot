---
name: verify
description: Build, launch and drive PX4 SITL headlessly to observe a change actually flying. Use when verifying multicopter control-chain changes (mc_controller, mc_att_control, mc_rate_control, mc_manual_mapping) or anything else observable from a running vehicle.
---

# Verifying a change by flying it in SITL

Surface = the running vehicle: MAVLink in, uORB topics and `px4-*` console
clients out. Not `make tests`.

## Build

```bash
make px4_sitl_default -j16      # ~10 min cold, seconds warm
```

## Launch headless — always use `-d`

`make px4_sitl gz_x500` redirected to a file **fills the disk**: with no TTY the
pxh console redraws its prompt in a loop (55 MB in ~4 min). Use daemon mode,
which runs no console:

```bash
cd build/px4_sitl_default/rootfs
PX4_SIM_MODEL=gz_x500 GZ_IP=127.0.0.1 HEADLESS=1 ../bin/px4 -d > boot.log 2>&1 &
# ready when: grep -q "Startup script returned successfully" boot.log
```

Then drive the console through the client binaries — they attach to the running
instance, so no TTY is needed:

```bash
build/px4_sitl_default/bin/px4-mc_controller status
build/px4_sitl_default/bin/px4-param show MC_CTRL_ALG
build/px4_sitl_default/bin/px4-listener vehicle_rates_setpoint
```

## Gotchas that cost real time here

- **`pkill -f "bin/px4"` kills your own shell** (the agent's command line
  contains the pattern) — the command dies with exit 144 and the sim survives.
  Kill by PID from `ps -eo pid,cmd`, filtering out your own script.
- Kill patterns must match **how the process was launched**. An instance started
  as `../bin/px4` does not match `px4_sitl_default/bin/px4`; the leftover makes
  the next boot fail with `PX4 server already running for instance 0`.
- One instance at a time — overlapping runs kill each other's simulator.
- Gz leaves a `gz sim ... -s` server behind after px4 exits. Kill it too.
- `libGstCameraSystem.so` load error at boot is pre-existing and harmless.

## Flying a profile

`src/modules/mc_controller/tools/flight_profile.py` (pymavlink, already
installed) flies arm → takeoff → position box → land over
`udp:127.0.0.1:14540`. Set `PROFILE_STAB=1` to also enter Stabilized, which is
the only segment reaching the Attitude level.

```bash
PROFILE_STAB=1 python3 src/modules/mc_controller/tools/flight_profile.py
```

Neither this profile nor the A/B tools ever reach **ACRO / BodyRate**. To cover
it: climb under OFFBOARD, stream `manual_control_send` *before* the mode switch,
switch to main mode 5, confirm engagement from HEARTBEAT
(`(custom_mode >> 16) & 0xFF == 5`) — the DO_SET_MODE ACK is not proof — then
read `vehicle_rates_setpoint` with `px4-listener`. At full stick,
`roll == radians(MC_ACRO_R_MAX)` exactly.

## Switching the controller arm

`MC_CTRL_ALG` is read by `rc.mc_apps` at boot, so it needs a param save and a
full restart, not just a `param set`:

```bash
px4-param set MC_CTRL_ALG 1 && px4-param save    # then restart px4
```

Always confirm the arm from `px4-mc_controller status` (`active controller:`),
never from the parameter alone. `0` = stock trio and `mc_controller` reports
"not running"; `1` = cascaded_pid; `2` = template; anything else logs
`not available, using reference cascade` and still flies.

## Comparing arms

`tools/{group_compare,pub_matrix}.py` take ULogs from
`build/px4_sitl_default/rootfs/log/<date>/*.ulg` (newest = the run you just
flew). `group_compare.py` needs **>=2 flights per arm** — on n=1 it tracebacks
with `max() iterable argument is empty` rather than saying so.

Neither segments by flight mode, so an error confined to one mode is diluted by
the whole-flight average. Window on the segment yourself before trusting a green
aggregate.
