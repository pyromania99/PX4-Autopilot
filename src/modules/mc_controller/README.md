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
    │ mc_controller        (rate_ctrl WQ, gyro rate)       │
    │   driven by vehicle_angular_velocity                 │
    │   VehicleStateProvider ──► ControllerState           │
    │   CommandFrontEnd      ──► ControllerCommand         │
    │        (level resolve, sticks, takeoff ramp,         │
    │         failsafe, EKF resets, limits)                │
    │                     │                                │
    │   MulticopterControllerBase*  ◄── MC_CTRL_ALG        │
    │     └─ TrajectoryStage (shared, opt-in)              │
    │                     │                                │
    │   OutputStage: NaN guard, watchdog, fallback         │
    └──────────────────────────────────────────────────────┘
                     │
        torque+thrust│
                     v
       control_allocator ──────────► MixingOutput ──► ESCs
```

The whole law runs on `rate_ctrl`; each stage keeps its own cadence by gating on
`state.freshness` rather than by splitting work queues — see
[One work queue](#one-work-queue).

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
| 3 | `EigenController` — eigen dynamics, cross-coupled roll/pitch rate law |

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

2. **Add parameters** to `module.yaml`, under the
   `Multicopter Controller Framework` group, and bind them with
   `DEFINE_PARAMETERS((ParamFloat<px4::params::MC_MY_GAIN>) _param_gain)`.
   Parameter refresh is automatic — `ModuleParams` cascades from the module.

   PX4 no longer scans `.c` files for `PARAM_DEFINE_*`: `px_process_params.py`
   reads only the generated params directory, which is built from the YAML
   registered through `MODULE_CONFIG`. A `params.c` here would be ignored
   silently.

3. **Add `MyController.cpp/.hpp` to `SRCS`** in `controllers/CMakeLists.txt`.

4. **Register it** in `controllers/ControllerRegistry.hpp/.cpp`: add an
   `Algorithm` enumerator and one `case`. `-Wswitch` will remind you if you add
   the enumerator and forget the case.

5. **Add the value** to `MC_CTRL_ALG` in `module.yaml` so QGC shows it.

Optionally, step 6: **reuse the shared trajectory stage** instead of writing your
own position loop — see below.

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

## Sharing the trajectory stage

Every law here needs the same thing between a position setpoint and an attitude
setpoint: a per-axis PD over whatever the flight mode actually commanded, a lateral
clamp bounded by the tilt limit, and a collective. That is `TrajectoryStage`, and
`CascadedPdController` and `EigenController` both use it.

**Opting in is composition.** Hold one, pass your own gains, delegate:

```cpp
#include <TrajectoryStage.hpp>

// ... a ModuleParams child, so its own gains refresh through the same cascade.
TrajectoryStage _trajectory_stage;   // constructed with (this)

void MyController::stepTrajectoryToAttitude(const ControllerState &state,
                                            const ControllerCommand &command)
{
    const Vector3f acceleration_sp =
        _trajectory_stage.computeAccelerationSetpoint(state, command, _pos_p, _pos_d);

    // ... your acceleration -> attitude conversion, which is YOUR law ...

    _thrust_setpoint = _trajectory_stage.computeThrustSetpoint(state, command, acceleration_sp);
}
```

**Opting out is not having one.** There is no virtual to override and no runtime
branch — write your own stage, as `TemplateController` does.

What it does NOT own is the acceleration → attitude conversion. That map is what
distinguishes the laws: `CascadedPdController` builds a body-z direction and a
quaternion, `EigenController` inverts to small-angle Euler scalars. Those are
different mathematical objects, not different spellings.

Three things to wire up if you use it:

- forward `command.reset_integrals` to `resetIntegral()`, and call `reset()` from
  your own `reset()`
- gate on `state.freshness.position_new || !_trajectory_stage.stageValid()`, and
  call `invalidateStage()` when you leave `Trajectory` level
- delegate `fillLocalPositionSetpoint()`, passing your attitude and thrust setpoints

### It owns the position/velocity integrator

`MC_OL_XY_I` / `MC_OL_Z_I`, bounded by `MC_OL_I_LIM`. They integrate the error
against the **equivalent velocity setpoint**, the same place stock `mc_pos_control`
integrates. These laws compute `pos_p*e_p + pos_d*(v_ff - v)`, which is
`pos_d*(v_sp - v)` with `v_sp = v_ff + (pos_p/pos_d)*e_p`, i.e. the stock cascade
written in one line. `limitVelocitySetpoint()` already builds that `v_sp`, and
the integrator runs on `v_sp - v` after the limit. On a hold axis that error is
`(pos_p/pos_d)*e_p`, so a standing offset is trimmed. On a velocity-only axis it
is the rate error, so a bias cannot swallow a stick command.

It used to integrate position error, which zeroed the term whenever the mode
commanded no position on that axis — ALTCTL and POSCTL with the throttle stick
deflected. With hover thrust wrong by 0.13, full stick-down then produced no
descent at all (2026-09-24 spin flights).

`MC_OL_Z_I` ships at **2.0**, stock's `MPC_Z_VEL_I_ACC`. `mc_hover_thrust_estimator`
normally supplies the vertical trim, but it only runs above 1 m `dist_bottom`, so
low indoor flight relies on this term. `MC_OL_XY_I` ships at **zero**: a PD law
holds position perfectly in simulation and settles at `disturbance / Kp` on
hardware — at `MC_EIG_XY_P = 0.5`, a 1 deg tilt bias is 0.34 m. Enabling it is a
hardware step. Note that no world-frame integrator can cancel a body-fixed lean
while the vehicle spins, since in the world frame that force rotates at the spin
rate.

## Roll/pitch torque trim (`TorqueTrim`)

This is the attitude-level integral, shared by `MC_CTRL_ALG=1` and `3` through
composition, like `RateLimits`. It removes a constant body-fixed torque: a CG offset or a
weak motor that a PD attitude law can only hold with a standing attitude error. It
**integrates the roll/pitch torque the law already commands**, not the attitude error,
and moves it into a trim. That converges whatever the loop's own gain and phase are.
While spinning, the loop rotates the attitude error relative to the torque by
`atan(ATT_D*r/ATT_P)` plus the gyroscopic coupling, and for the eigen law by the seat
angle as well, and an attitude-error integral diverges once that passes 90°.
`TorqueTrimTest` asserts both halves of that claim.

It is applied last, after the seat and the actuator lead, and before the output clamp.
It trims roll and pitch only, freezes on allocator saturation and at the clamp, and
resets with `reset_integrals`. `MC_ATT_TRIM_I` ships at 0 and `MC_ATT_TRIM_LIM` at 0.1.
It is logged as `mc_controller_status.torque_trim`, and for alg 1 also in
`rate_ctrl_status`.

## One work queue

Stock PX4 splits the cascade across two work queues: `mc_pos_control` on
`nav_and_controllers` at ~125 Hz, `mc_rate_control` on the high-priority `rate_ctrl`
queue at gyro rate. Keeping heavy trajectory math off the fast queue is what protects
inner-loop timing on flight hardware.

**Here the whole law runs on `rate_ctrl`, and stage-rate parity comes from freshness
gating instead.** `update()` is called at gyro rate, but each stage only does work when
its input is new:

```cpp
if (state.freshness.position_new || !_trajectory_stage.stageValid()) {
    stepTrajectoryToAttitude(state, command);   // position rate
}
```

Measured in SITL: 3585 position-stage runs against 7171 cycles, exactly the 1:2 the
250 Hz gyro / 125 Hz position rates imply. The gating is rate-agnostic — on an 800 Hz
gyro with position at 50 Hz it is 1:16, with no code change — and `dt_position` remains
the true interval between the two samples the error was measured across. A fixed
divider would not survive that, because its `dt` would drift against real arrivals.

| stage | runs at | gate |
|---|---|---|
| position → acceleration → collective | position rate | `freshness.position_new` |
| attitude-error derivative | attitude rate | `freshness.attitude_new` |
| rate loop → torque | gyro rate | every cycle |

The framework once carried an optional second work item (`OuterLoop`) that moved the
trajectory stage to `nav_and_controllers` behind a `hasOuterStage()` opt-in. No shipped
law ever opted in, so it woke at position rate purely to return immediately, and it was
removed along with its four virtuals and the cross-thread contract they required. If a
future law genuinely needs the split — an outer stage too expensive for the gyro path —
it is reintroducible, but freshness gating is the cheap answer and should be tried first.

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
| position/velocity PD, position integrator, lateral clamp, collective | `TrajectoryStage`, if you opt in |
| stick → attitude and stick → rate mapping | `src/lib/mc_manual_mapping` |
| hover-thrust estimate, throttle slews, spool-up gating | front end + state provider |
| battery scaling, NaN guard, watchdog, fallback latch | output stage |
| `vehicle_attitude_setpoint` / `vehicle_rates_setpoint` / `vehicle_local_position_setpoint` publication | output stage |

Publication is **level-gated to match stock exactly**: a topic is published only
in the modes where the stock module that owns it actually runs, and only by the
work item that generated it. `vehicle_local_position_setpoint` therefore appears
only at `Trajectory` level, from the output stage. Getting this wrong does not disturb flight — it
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
