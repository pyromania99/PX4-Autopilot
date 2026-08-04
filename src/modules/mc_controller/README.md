# mc_controller — pluggable multicopter controller framework

> **[TESTING.md](TESTING.md)** — **add a controller and prove it works.** The three
> files you write, the four one-line edits, and the four testing rungs from unit
> test to hardware. Start here if you want to get your own law flying.
>
> **[CHANGE_MAP.md](CHANGE_MAP.md)** — diagrams: architecture before/after, every
> file touched, one control cycle, build order. Start here if you want the shape.
>
> **[ARCHITECTURE.md](ARCHITECTURE.md)** — the full narrative: PX4's original
> control architecture, why swapping a controller was hard, every change, and the
> stage-by-stage verification record including the defects it caught.

A black box that takes the EKF state plus the commanded setpoint and produces
motor moments, selected at runtime by `MC_CTRL_ALG`. Built for fault-tolerant
control research: swap in an INDI / LQR / SE(3) law without forking
`mc_pos_control`, `mc_att_control` and `mc_rate_control`.

```
flight_mode_manager ──trajectory_setpoint──┐
manual_control_setpoint ───────────────────┤
vehicle_control_mode / status / land ──────┤
local_position / attitude / angular_vel ───┤
                                           v
    ┌──────────────────────────────────────────────────────┐
    │ mc_controller_outer  (nav_and_controllers, ~125 Hz)  │
    │   driven by vehicle_local_position                   │
    │   VehicleStateProvider + CommandFrontEnd (its own)   │
    │   controller->updateOuter()   ── the heavy           │
    │                                  trajectory math     │
    └──────────────────────────────────────────────────────┘
           │ vehicle_attitude_setpoint (uORB, no shared state)
           v
    ┌──────────────────────────────────────────────────────┐
    │ mc_controller        (rate_ctrl WQ, gyro rate)       │
    │   driven by vehicle_angular_velocity                 │
    │   VehicleStateProvider ──► ControllerState           │
    │   CommandFrontEnd      ──► ControllerCommand         │
    │        (level resolve, sticks, takeoff ramp,         │
    │         failsafe, EKF resets, limits)                │
    │                     │                                │
    │   MulticopterControllerBase*  ◄── MC_CTRL_ALG        │
    │                     │                                │
    │   OutputStage: NaN guard, watchdog, fallback         │
    └──────────────────────────────────────────────────────┘
                     │
        torque+thrust│
                     v
       control_allocator ──────────► MixingOutput ──► ESCs
```

The outer item exists **only if your controller opts in** with `hasOuterStage()`.
A monolithic full-stack law leaves it off and runs entirely on `rate_ctrl` — see
[Two work queues, or one](#two-work-queues-or-one).

## Quick start

```
param set MC_CTRL_ALG 1     # 1 = cascaded PD
reboot                      # 0 <-> non-zero changes which modules start
mc_controller status
```

`MC_CTRL_ALG=0` (the default) does **not** start this module — the stock chain
runs instead. Nothing changes until you deliberately enable it.

| value | controller |
|---|---|
| 0 | stock `mc_pos_control` + `mc_att_control` + `mc_rate_control` |
| 1 | `CascadedPdController` — cascaded PD with a geometric attitude law |
| 2 | `TemplateController` — skeleton to copy |

The controllers under `controllers/` are **independent implementations**, not
re-wrappings of the stock cascade: they do not link `PositionControl`,
`AttitudeControl` or `RateControl`, and they do not read the `MPC_` / `MC_` gains.
Reaching the stock cascade means `MC_CTRL_ALG=0` and a reboot, because that is a
decision about which modules the startup script launches.

### `MC_CTRL_ALG=1` does not control yaw

`CascadedPdController` builds its attitude setpoint around the vehicle's *current*
heading every cycle, and the yaw axis gets rate damping only (`MC_PD_YAWR_D`,
default `0`). RC yaw stick and mission yaw setpoints therefore do nothing, and
heading free-runs. That is the design: a multirotor that has lost a rotor cannot
hold heading, and surrendering yaw is what leaves enough authority to hold
position.

It is also the framework's always-allocated failsafe (`_reference`), so a fallback
latch lands here rather than on anything stock.

## Adding a controller — 5 steps

1. **Copy the template.**
   ```
   cp controllers/TemplateController.{hpp,cpp} controllers/MyController.{hpp,cpp}
   ```
   Rename the class, and return your own `name()`.

2. **Add parameters** in `controllers/my_controller_params.c` using
   `PARAM_DEFINE_FLOAT(MC_MY_GAIN, ...)`, and bind them with
   `DEFINE_PARAMETERS((ParamFloat<px4::params::MC_MY_GAIN>) _param_gain)`.
   Parameter refresh is automatic — `ModuleParams` cascades from the module.

3. **Add `MyController.cpp/.hpp` to `SRCS`** in `controllers/CMakeLists.txt`.
   Do **not** add the `_params.c` file: `src/lib/parameters/CMakeLists.txt`
   glob-recurses `src/*params.c` and generates the definitions itself, so
   compiling it into the library as well is a redefinition error.

4. **Register it** in `controllers/ControllerRegistry.hpp/.cpp`: add an
   `Algorithm` enumerator and one `case`. `-Wswitch` will remind you if you add
   the enumerator and forget the case.

5. **Add the value** to `MC_CTRL_ALG` in `module.yaml` so QGC shows it.

## The interface

```cpp
class MyController : public MulticopterControllerBase
{
    const char *name() const override { return "mine"; }

    // Which levels you handle. Full stack = mc_ctrl::kAllLevels.
    uint8_t supportedLevels() const override { return mc_ctrl::kAllLevels; }

    // Zero every integrator / filter / internal state.
    void reset() override;

    bool update(const mc_ctrl::ControllerState &state,
                const mc_ctrl::ControllerCommand &command,
                float dt,
                mc_ctrl::ControllerOutput &output) override;
};
```

`command.level` tells you what was commanded:

| level | what you get | flight modes |
|---|---|---|
| `Trajectory` | `position_sp` / `velocity_sp` / `acceleration_sp` / `yaw_sp` in NED, plus the limit envelope | Position, Altitude, Auto, Offboard pos/vel/accel, Orbit |
| `Attitude` | `attitude_sp` (unit quaternion), `yaw_sp_move_rate`, `thrust_body_sp` | Manual, Stabilized, Offboard attitude |
| `BodyRate` | `rate_sp` (guaranteed finite), `thrust_body_sp` | Acro, Offboard body_rate |
| `None` | nothing — return `false` | Termination, disarmed, not rotary wing |

This is what lets one control law serve every flight mode: you never touch
`vehicle_control_mode` or worry about which mode is active.

## Two work queues, or one

Stock PX4 splits the cascade across two work queues: `mc_pos_control` on
`nav_and_controllers` at ~125 Hz, `mc_rate_control` on the high-priority
`rate_ctrl` queue at gyro rate. Keeping heavy trajectory math off the fast queue
is what protects inner-loop timing on flight hardware.

**The default here is one queue.** Your `update()` runs the whole law at gyro
rate. That is the right choice for a monolithic controller — an INDI or SE(3) law
whose stages share state cannot be cut in half.

If your law *does* separate cleanly, opt in:

```cpp
bool hasOuterStage() const override { return true; }

bool updateOuter(const mc_ctrl::ControllerState &state,
                 const mc_ctrl::ControllerCommand &command,
                 float dt, vehicle_attitude_setpoint_s &attitude_setpoint) override;
```

`updateOuter()` then runs on `nav_and_controllers` driven by
`vehicle_local_position`; its published `vehicle_attitude_setpoint` reaches
`update()` as `command.attitude_sp` with `command.outer_stage_complete == true`,
and `update()` skips its own trajectory stage.

> **The cross-thread contract.** `updateOuter()` and `update()` run
> **concurrently on different work queues, on the same controller object.** They
> MUST touch disjoint member state. Anything that has to cross between them
> travels through the published `vehicle_attitude_setpoint`, not a shared member.
> There is no lock, deliberately — a mutex on the gyro path is worse than the
> latency it would remove.

Neither shipped controller opts in today. `CascadedPdController` deliberately does
not: the whole law is a few dozen flops, so paying for the state-partitioning
contract buys nothing, and its position stage still runs at position rate because
`update()` gates it on `state.freshness.position_new`. **Gating on freshness, not
splitting the queue, is the cheap way to get stage-rate parity** — reach for
`hasOuterStage()` only when the outer stage is genuinely too expensive for the
gyro path.

The split path is still live and exercised by `OuterLoop`; measured placement when
a controller does opt in, against stock:

```
STOCK   wq:rate_ctrl            mc_rate_control       250.0 Hz
        wq:nav_and_controllers  mc_att_control        250.0 Hz
                                mc_pos_control        125.0 Hz
SPLIT   wq:rate_ctrl            mc_controller         250.0 Hz
        wq:nav_and_controllers  mc_controller_outer   125.0 Hz
```

**Known deviation from stock:** the attitude stage sits on `rate_ctrl` here, not
on `nav_and_controllers`. Deliberate — attitude is a cheap quaternion P law, and
pairing it with the rate loop removes a uORB hop of latency from the fast
cascade; the position stage is the expensive part that needed to move. Exact stock
placement would need a third work item.

## Rules that will bite you

- **Honour `command.reset_integrals`.** The framework cannot reach into your
  integrators. It is set on mode change, on the ground and when disarmed. Ignore
  it and you wind up while sitting on the ground, then take off violently.
- **Always fill `output.thrust`, not just `output.torque`.** `land_detector` and
  `mc_hover_thrust_estimator` consume the published `vehicle_thrust_setpoint`;
  leaving it NaN breaks land detection so auto-land never disarms. The output
  stage rejects it, but as a fallback, not as flow control.
- **`update()` runs at gyro rate on a real-time work queue.** No heap
  allocation, no blocking, no `printf`, bounded execution time.
- **Declare every level you handle.** Entering an undeclared level latches the
  reference-cascade fallback, logs loudly, and is reported at boot.
- **NaN is not a safe default.** Downstream of `control_allocator`, `NaN` in
  `actuator_motors.control` means *stop that motor*. The output stage rejects
  non-finite output and falls back, but do not rely on it as flow control.

## What the framework does for you

You do not reimplement, and must not fight, any of this:

| behaviour | where |
|---|---|
| takeoff ramp + `takeoff_status` | `CommandFrontEnd` (delivered as `command.vel_limit_up`) |
| two-stage failsafe (200 ms last-valid, then blind descend) | `CommandFrontEnd` |
| EKF reset application to setpoints | `CommandFrontEnd` |
| velocity notch / low-pass filter chain | `VehicleStateProvider` |
| stick → attitude and stick → rate mapping | `src/lib/mc_manual_mapping` |
| hover-thrust estimate, throttle slews, spool-up gating | front end + state provider |
| battery scaling, NaN guard, watchdog, fallback latch | output stage |
| `vehicle_attitude_setpoint` / `vehicle_rates_setpoint` / `vehicle_local_position_setpoint` publication | output stage + `OuterLoop` |

Publication is **level-gated to match stock exactly**: a topic is published only
in the modes where the stock module that owns it actually runs, and only by the
work item that generated it. `vehicle_local_position_setpoint` therefore appears
only at `Trajectory` level, from `OuterLoop` when the outer stage is active and
from the output stage otherwise. Getting this wrong does not disturb flight — it
feeds the flight tasks a stale reset origin and makes log analysis lie (it
produced a phantom 12× altitude-tracking regression; see Stage 12 in the smoke
log).

## Output contract

There is exactly one: publish normalized `torque[3]` + `thrust[3]`, and
`control_allocator` mixes them, giving you `CA_*` geometry, motor-failure
handling, slew limits and desaturation for free.

A controller does **not** publish `actuator_motors` itself. `control_allocator`
publishes that topic unconditionally on every cycle and is backup-scheduled at
20 Hz even when no torque setpoint arrives, so a second publisher does not take
ownership — it contends, and the mixer sees whichever message landed last.

For failure-tolerant allocation, the extension point is `control_allocator`'s
effectiveness matrix (`CA_METHOD`, `CA_FAILURE_MODE`), not this interface.

## Diagnostics

```
mc_controller status          # active controller, level, fallback state, counters
listener mc_controller_status # torque/thrust/motors, per-stage dt, fallback reason
```

`mc_controller_status.debug[8]` is free for your own use — fill it from
`fillStatus()`.

## Verification

Every stage of this framework was gated; the record including every failure found
is in [`src/lib/mc_manual_mapping/SMOKE_LOG.md`](../../lib/mc_manual_mapping/SMOKE_LOG.md).

To re-check the framework after changing anything:

```
make tests TESTFILTER=CascadedPd
make tests TESTFILTER=ControlLevelResolver
make tests TESTFILTER=CommandFrontEnd
make tests TESTFILTER=VehicleState
make tests TESTFILTER=ControllerApi
```

Note what these can and cannot tell you. Everything below the controller — level
resolution, state assembly, the command envelope, the fallback machinery — is
still verified against the behaviour it was ported from. The **controller** is
not: `CascadedPdController` is an independent control law, so its test asserts the
properties the law is supposed to have (equilibrium, sign, saturation, tilt
limit, yaw inaction, fail-closed on NaN) rather than agreement with anything.
Whether the gains actually fly is a SITL and flight-test question, not a unit-test
one — start from `tools/flight_profile.py`.
