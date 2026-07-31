# Pluggable Multicopter Controller Framework — Architecture, Changes and Verification

> Prefer diagrams? [**CHANGE_MAP.md**](CHANGE_MAP.md) has the same material as four
> flowcharts — architecture before/after, every file touched, one control cycle,
> and the gated build order.

This document assumes no prior knowledge of the work. It covers:

1. [What PX4's multicopter control stack looked like before](#1-the-original-architecture)
2. [Why swapping in a new controller was hard](#2-why-swapping-a-controller-was-hard)
3. [What was built](#3-the-new-architecture)
4. [Complete file inventory](#4-file-inventory)
5. [Stage-by-stage: changes, tests, and what each found](#5-stage-by-stage-record)
6. [Full test inventory](#6-test-inventory)
7. [Known limitations and what remains unverified](#7-known-limitations-and-unverified-items)
8. [How to use it](#8-how-to-use-it)

Two facts to hold onto while reading:

- **`MC_CTRL_ALG` defaults to `0`.** With that default the module is not even
  started and the stock control chain runs exactly as before. None of this
  changes vehicle behaviour until deliberately enabled.
- **Nothing is committed.** All of it sits in the working tree of branch
  `motor_failure` for review.

---

## 1. The original architecture

### 1.1 The signal chain

PX4 flies a multicopter through a cascade of three separate modules, each a
separate process-like work-queue item communicating only over uORB (PX4's
publish/subscribe bus):

```
flight_mode_manager
    │  trajectory_setpoint          (position/velocity/accel/yaw in NED)
    ▼
mc_pos_control                       ~100 Hz, "nav_and_controllers" work queue
    │  vehicle_attitude_setpoint     (quaternion q_d + thrust_body)
    ▼
mc_att_control                       ~250 Hz, "nav_and_controllers" work queue
    │  vehicle_rates_setpoint        (roll/pitch/yaw rates + thrust_body)
    ▼
mc_rate_control                      ~250 Hz (gyro rate), "rate_ctrl" work queue
    │  vehicle_torque_setpoint       (normalized torque, body FRD)
    │  vehicle_thrust_setpoint       (normalized thrust, body FRD)
    ▼
control_allocator                    "rate_ctrl" work queue
    │  actuator_motors               (normalized per-motor, 12 channels)
    ▼
MixingOutput  (src/lib/mixer_module)
    │  actuator_outputs
    ▼
output drivers (pwm_out, dshot, uavcan, sim, …) → ESCs
```

Each stage is driven by a *different* uORB callback, so they run at different
rates on two different work queues:

| module | driven by | work queue | dt clamp |
|---|---|---|---|
| `mc_pos_control` | `vehicle_local_position` | `nav_and_controllers` | 0.002 – 0.04 s |
| `mc_att_control` | `vehicle_attitude` | `nav_and_controllers` | 0.0002 – 0.02 s |
| `mc_rate_control` | `vehicle_angular_velocity` | `rate_ctrl` | 0.000125 – 0.02 s |

That detail matters later: it means there is a real uORB publish→subscribe hop
between the attitude loop and the rate loop, worth roughly one cycle of latency.

### 1.2 Where the control mathematics actually lives

The three modules are mostly *plumbing*. The control laws are in small,
self-contained classes:

| class | file | what it does |
|---|---|---|
| `PositionControl` | `src/modules/mc_pos_control/PositionControl/` | P on position → PID on velocity → acceleration → thrust vector, tilt limiting, hover-thrust scaling; `ControlMath::thrustToAttitude()` converts the thrust vector to an attitude setpoint |
| `AttitudeControl` | `src/modules/mc_att_control/AttitudeControl/` | reduced-attitude quaternion proportional law (Brescianini/Hehn/D'Andrea), `MC_YAW_WEIGHT` yaw de-prioritisation, rate saturation |
| `RateControl` | `src/lib/rate_control/` | `torque = Kp·e + I − Kd·ω̇ + Kff·ω_sp` — note the D term acts on *measured angular acceleration*, not error derivative |
| `TakeoffHandling` | `src/modules/mc_pos_control/Takeoff/` | smooth-takeoff state machine and climb-rate ramp |
| `GotoControl` | `src/modules/mc_pos_control/GotoControl/` | goto-setpoint smoothing |

**None of these classes have any virtual methods.** They are concrete types
instantiated as direct members of their owning module.

### 1.3 How flight modes select which loops run

There is no explicit "mode" input to the controllers. Instead `commander`
publishes `vehicle_control_mode`, a set of boolean flags, and each module decides
for itself whether to run:

- **`mc_pos_control`** runs when `flag_multicopter_position_control_enabled`.
  That flag is derived in `Commander.cpp:2652` as
  `rotary_wing && (altitude || climb_rate || position || velocity || acceleration)`.
- **`mc_att_control`** runs its attitude loop when
  `flag_control_attitude_enabled && (hovering || tailsitter_in_transition)`, and
  additionally *generates* an attitude setpoint from RC sticks when
  `manual && !altitude && !velocity && !position` (`mc_att_control_main.cpp:297-300`).
- **`mc_rate_control`** runs the rate loop when `flag_control_rates_enabled`, and
  generates a rate setpoint from sticks (Acro) when `manual && !attitude`
  (`MulticopterRateControl.cpp:156`).

So the "what level am I being commanded at" decision is **implicit and split
across three separate files**.

### 1.4 Safety behaviour embedded in the modules

Beyond control mathematics, `mc_pos_control`'s `Run()` loop also owns a
substantial amount of safety-critical policy:

- smooth-takeoff ramp expressed as a climb-rate limit
- a two-stage failsafe: reuse the last valid setpoint for up to 200 ms, then
  generate a blind stop/descend setpoint
- applying EKF reset deltas to in-flight setpoints so a GNSS jump does not step
  the setpoint
- a velocity notch → low-pass → derivative filter chain
- an on-ground override that commands 100 m/s² *downward* acceleration so no
  thrust is produced before takeoff, plus integrator resets
- tilt-limit slewing from `MPC_TILTMAX_LND` to `MPC_TILTMAX_AIR`
- hover-thrust-estimate integration

`mc_att_control` similarly owns stick→tilt mapping, `StickYaw` heading handling,
throttle curves, spool-up slews, EKF quaternion-reset handling and autotune
injection. `mc_rate_control` owns the Acro stick shaping, allocator anti-windup,
a yaw-torque low-pass filter and battery scaling.

---

## 2. Why swapping a controller was hard

The goal was a *black box*: something that takes the estimator state plus
whatever the flight mode commands, and produces motor moments — so a
fault-tolerant control law (INDI, LQR, SE(3), a learned policy) can replace the
cascade without forking PX4.

Four obstacles:

1. **No interface exists.** `PositionControl`, `AttitudeControl` and
   `RateControl` have no virtual methods, and there is no parameter selecting
   among control strategies. Replacing the cascade meant forking three modules.

2. **The safety policy is tangled with the control law.** All of §1.4 lives
   inside module `Run()` loops. A research controller replacing those modules
   would have to reimplement the takeoff ramp, the failsafe and the EKF-reset
   bookkeeping — safety-critical code that is easy to get subtly wrong.

3. **Mode gating is implicit and triplicated.** A replacement has to reproduce
   three separate gating predicates exactly, or it will run at the wrong level.

4. **The only existing precedents don't generalise.** `mc_nn_control` and
   `mc_raptor` bypass the cascade by publishing `actuator_motors` directly, but
   they only get away with silencing `control_allocator` because they register as
   *external flight modes*. That mechanism has no effect on built-in POSCTL /
   ALTCTL / AUTO modes, which is where you actually fly.

There is, however, good precedent for the *pattern* of a runtime-selected
strategy: `ControlAllocation` is an abstract base with implementations chosen by
`CA_METHOD`, and `ActuatorEffectiveness` likewise by `CA_AIRFRAME`. This work
follows that pattern.

---

## 3. The new architecture

### 3.1 Overview

A single new module, `mc_controller`, replaces all three stock control modules
when enabled. `flight_mode_manager` keeps running unchanged — it is the setpoint
*generator* (flight tasks, collision prevention, smooth trajectories), not a
controller, and replacing it would duplicate thousands of lines for no benefit.

```
flight_mode_manager ──trajectory_setpoint / vehicle_constraints──┐
manual_control_setpoint ─────────────────────────────────────────┤
vehicle_control_mode / vehicle_status / land_detected ───────────┤
local_position / attitude / angular_velocity / hover_thrust ─────┤
                                                                 ▼
        ┌────────────────────────────────────────────────────────────┐
        │ mc_controller        WorkItem, "rate_ctrl" WQ,             │
        │                      driven by vehicle_angular_velocity    │
        │                                                            │
        │   VehicleStateProvider ────► ControllerState               │
        │       EKF topics → filtered state + reset deltas           │
        │                                                            │
        │   CommandFrontEnd      ────► ControllerCommand             │
        │       level resolution, stick mapping, takeoff ramp,       │
        │       failsafe, EKF-reset application, limit envelope      │
        │                             │                              │
        │   MulticopterControllerBase*│ ◄── selected by MC_CTRL_ALG  │
        │                             │                              │
        │   Output stage                                             │
        │       NaN guard, watchdog, battery scale, fallback latch,  │
        │       and the ONLY place that publishes motor commands     │
        └────────────────────────────────────────────────────────────┘
              │                                    │
  torque+thrust│                                   │actuator_motors (direct)
              ▼                                    ▼
       control_allocator ───────────────────► MixingOutput ──► ESCs
```

Since Stage 12 there is a **second, optional work item**. Stock PX4 spreads the
cascade over two work queues — `mc_pos_control` on `nav_and_controllers` at
~125 Hz, `mc_rate_control` on the high-priority `rate_ctrl` queue at gyro rate —
so that heavy trajectory math never delays the inner loop. Running everything on
`rate_ctrl`, as the design originally did, put `PositionControl` on the fast
queue. `OuterLoop` restores the separation:

```
        ┌────────────────────────────────────────────────────────────┐
        │ OuterLoop      WorkItem, "nav_and_controllers" WQ,         │
        │                driven by vehicle_local_position (~125 Hz)  │
        │   its OWN VehicleStateProvider + CommandFrontEnd           │
        │   controller->updateOuter()  ──► vehicle_attitude_setpoint │
        └────────────────────────────────────────────────────────────┘
                      │ vehicle_attitude_setpoint  (uORB)
                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │ MulticopterController   "rate_ctrl" WQ, gyro rate          │
        │   sees command.outer_stage_complete == true and skips its   │
        │   own trajectory stage                                     │
        └────────────────────────────────────────────────────────────┘
```

This is **opt-in per controller**, and that is a requirement rather than a
convenience. The framework's scope is full-stack monolithic control laws — an INDI
or SE(3) controller whose position, attitude and rate stages share internal state
cannot be cut across two threads. Such a controller leaves `hasOuterStage()`
false, `OuterLoop` returns immediately every cycle, and the whole law runs at gyro
rate exactly as before.

The two items own **separate** `VehicleStateProvider` and `CommandFrontEnd`
instances. Sharing either would be a data race across work queues. The only
object they share is the controller itself, whose two entry points are
contractually required to touch disjoint members — everything that must cross
between them travels through the published `vehicle_attitude_setpoint`. There is
no mutex, deliberately: a lock on the gyro path costs more than the latency it
removes.

### 3.2 The central idea: an explicit control level

The design's key move is to make the implicit "what am I being commanded at"
decision **explicit and single-sourced**:

```cpp
enum class ControlLevel : uint8_t {
    None       = 0,   // termination, disarmed, not rotary wing
    BodyRate   = 1,   // rate_sp + thrust_body_sp        (Acro, offboard body_rate)
    Attitude   = 2,   // attitude_sp + thrust_body_sp    (Manual, Stabilized, offboard attitude)
    Trajectory = 3,   // position/velocity/accel/yaw NED (Altitude, Position, Auto, Offboard, Orbit)
};
```

One function resolves it (`CommandFrontEnd/ControlLevelResolver.cpp`):

```cpp
ControlLevel resolveLevel(const vehicle_control_mode_s &vcm, uint8_t vehicle_type,
                          bool in_transition, bool is_tailsitter)
{
    if (vcm.flag_control_termination_enabled || !vcm.flag_control_rates_enabled) {
        return ControlLevel::None;                        // mirrors mc_rate_control's gate, inverted
    }
    if (vcm.flag_multicopter_position_control_enabled) {
        return ControlLevel::Trajectory;                  // mc_pos_control's own run condition
    }
    const bool hovering = (vehicle_type == ROTARY_WING) && !in_transition;
    const bool tailsitter_transition = is_tailsitter && in_transition;
    if (vcm.flag_control_attitude_enabled && (hovering || tailsitter_transition)) {
        return ControlLevel::Attitude;                    // mc_att_control's gate
    }
    return ControlLevel::BodyRate;                        // Acro / offboard body_rate
}
```

A controller therefore never inspects `vehicle_control_mode` and never asks which
flight mode is active. It receives a `ControllerCommand` tagged with the level and
handles the levels it declares support for.

**Why this is equivalent to stock**, which is the design's main correctness claim:

- `flag_multicopter_position_control_enabled` *is* `mc_pos_control`'s run
  condition, so `Trajectory` covers exactly the modes stock position control
  serves.
- `mc_att_control`'s stick gate `manual && !altitude && !velocity && !position`
  is identical to `level == Attitude && manual`, because no manual multicopter
  mode sets `acceleration` or `climb_rate` without also setting `altitude`.
- `mc_rate_control`'s acro gate `manual && !attitude` is identical to
  `level == BodyRate && manual`.

These are not asserted in prose only — they are checked per-`nav_state` in a test
that links commander's own mode table (see §6).

### 3.3 The interface

Three plain structs and one abstract class, in `ControllerApi/`.

**`ControllerState`** — everything the estimator knows. Frames: translational NED
(m, m/s, m/s²); rotational body FRD (rad/s, rad/s²); `q` is a Hamilton quaternion
rotating body-FRD → NED. NaN in position/velocity/acceleration means "this axis is
not estimable".

```cpp
struct ControllerState {
    uint64_t timestamp_sample;
    matrix::Quatf    q;
    matrix::Vector3f angular_velocity, angular_accel;
    matrix::Vector3f position, velocity, acceleration;   // filtered
    float heading, unaided_heading;
    bool position_valid_xy, position_valid_z, velocity_valid_xy, velocity_valid_z;
    EkfResets      resets;        // one-shot deltas since the previous cycle
    StateFreshness freshness;     // per-stage dt + *_new flags
    bool landed, maybe_landed, ground_contact, freefall, armed, spooled_up;
    float hover_thrust; bool hover_thrust_valid;
};
```

`StateFreshness` is what preserves stock's multi-rate behaviour. The framework
runs at gyro rate, but position and attitude samples arrive more slowly, so it
carries `dt`, `dt_attitude`, `dt_position` and `position_new` / `attitude_new`
flags. A controller wanting stage-wise rate parity with stock gates on those.

**`ControllerCommand`** — the setpoint, at the level indicated. NaN means "not
commanded" throughout. The framework guarantees: at `BodyRate`, `rate_sp` is fully
finite (NaN axes replaced with measured rate); at `Attitude`, `attitude_sp` is a
finite unit quaternion; at `Trajectory`, per-axis NaN semantics identical to
`trajectory_setpoint`. It also carries the limit envelope the framework computed
(`vel_limit_xy/up/down`, `thrust_min/max`, `tilt_limit`) and one hard contract:

```cpp
bool reset_integrals;   // set on mode change, on the ground, and when disarmed
```

The framework cannot reach into a plugin's integrators, so this is the single
behaviour that depends on controller cooperation. Ignoring it means winding up on
the ground and then taking off violently.

**`ControllerOutput`** — deliberately **fails closed**:

```cpp
struct ControllerOutput {
    matrix::Vector3f torque{NAN, NAN, NAN};
    matrix::Vector3f thrust{NAN, NAN, NAN};   // REQUIRED, not just torque
    matrix::Quatf    attitude_setpoint{NAN,NAN,NAN,NAN};  // optional, republished
    matrix::Vector3f rate_setpoint{NAN,NAN,NAN};          // optional, republished
    bool valid{false};
};
```

A controller that forgets to write its output is *rejected* rather than silently
commanding zero torque — which on a multicopter is free fall.

`thrust` being mandatory alongside `torque` is not stylistic: `land_detector` and
`mc_hover_thrust_estimator` both subscribe to the published
`vehicle_thrust_setpoint`. A controller that fills only torque flies fine and then
**never disarms after landing**.

**`MulticopterControllerBase`**:

```cpp
class MulticopterControllerBase : public ModuleParams
{
    virtual const char *name() const = 0;
    virtual uint8_t supportedLevels() const = 0;         // bitmask of levelBit(...)
    virtual void reset() = 0;                            // zero all internal state
    virtual bool update(const ControllerState &, const ControllerCommand &,
                        float dt, ControllerOutput &) = 0;
    virtual void setAllocatorFeedback(const AllocatorFeedback &);   // optional anti-windup
    virtual void fillStatus(mc_controller_status_s &) const;        // optional diagnostics
    virtual int  printStatus() const;
};
```

Deriving from `ModuleParams` gives every controller `DEFINE_PARAMETERS(...)` and
automatic parameter refresh via the parent→child cascade — no plumbing needed.

`supportedLevels()` is a bitmask rather than a single value so a *partial-stack*
controller is expressible: a rate-only fault-tolerant allocator returns just
`levelBit(BodyRate)`. Entering an undeclared level latches a fallback to the
reference cascade, logs loudly, and is reported at boot rather than discovered in
the air.

### 3.4 Division of responsibility

The framework deliberately keeps everything safety-critical:

| behaviour | owner | why |
|---|---|---|
| takeoff ramp + `takeoff_status` | framework | feeds `land_detector`; reaches the controller as `command.vel_limit_up` |
| two-stage failsafe (200 ms last-valid, then blind descend) | framework | setpoint *policy*, not a control law |
| EKF reset application to setpoints | framework | pure estimator bookkeeping |
| velocity notch / low-pass / derivative filters | framework | `VehicleStateProvider` |
| stick → attitude and stick → rate mapping | framework | shared library, see §3.6 |
| hover-thrust estimate, throttle slews, spool-up gating | framework | `COM_SPOOLUP_TIME` is an arming-safety guarantee |
| battery scaling (`MC_BAT_SCALE_EN`) | framework | identical post-processing for all controllers |
| NaN rejection, watchdog, fallback latch | framework | a controller cannot police itself |
| `vehicle_attitude_setpoint` / `vehicle_rates_setpoint` republication | framework | WeatherVane, mavlink, gimbal consume them |
| **integrator reset** | **shared** | framework sets `command.reset_integrals`; the controller must honour it |

### 3.5 The output stage — the safety boundary

Every cycle, after the controller returns:

1. If `!valid` or any non-finite value → latch fallback, reason
   `INVALID_OUTPUT`, `mavlink_log_critical`, and **re-run the reference cascade in
   the same cycle** so the vehicle is never left without a command.
2. If the active level is outside `supportedLevels()` → latch fallback, reason
   `UNSUPPORTED_LEVEL`.
3. Sanitise NaN → 0 (mirroring stock's guard), apply battery scaling, constrain.
4. Publish `vehicle_thrust_setpoint` + `vehicle_torque_setpoint`.
5. Republish the intermediate setpoints, `rate_ctrl_status`,
   `actuator_controls_status_0`, `takeoff_status`.
6. Publish `mc_controller_status` at ~50 Hz.

Two properties worth stating:

- **The fallback latch is sticky for the armed period**, cleared only on the
  armed→disarmed *transition*. A NaN means corrupted internal state; retrying next
  cycle would produce a limit cycle alternating between garbage and fallback.
- **The reference cascade is always allocated**, so fallback is instantaneous and
  allocation-free. When `MC_CTRL_ALG=1` the active controller *aliases* it and the
  destructor guards against a double free. `MC_CTRL_ALG` changes never allocate
  while armed — a mid-flight change is deferred to the next disarm with a warning,
  because `new`/`delete` on the real-time path risks heap fragmentation.

### 3.6 Shared stick mapping

The stick→setpoint mappings were **extracted verbatim** out of the stock modules
into `src/lib/mc_manual_mapping/`:

- `StickToAttitudeSetpoint` ← `MulticopterAttitudeControl::generate_attitude_setpoint()`
  plus `throttle_curve()` and their state (`StickYaw`, tilt input filters, throttle
  slew rates)
- `StickToRateSetpoint` ← the Acro superexpo mapping from `MulticopterRateControl`

Both stock modules were refactored to call the library, so there is exactly **one**
implementation. The alternative — copying it into the framework — would let the
two copies drift the first time upstream touches `MPC_THR_CURVE`, and would create
two implementations of `MPC_MANTHR_MIN` ramping and `COM_SPOOLUP_TIME` gating,
which are safety behaviour.

Net effect on the stock modules: **−205 lines, +19 lines**, a pure relocation.

Notably, **no parameter definitions moved.** Definitions are already scattered
(`MC_AIRMODE` is defined in `src/lib/mixer_module/params.c`, `MPC_MAN_TILT_MAX` in
`mc_pos_control`, `MC_ACRO_*` in `mc_rate_control`) and `DEFINE_PARAMETERS` binds
purely by name. Leaving them in place made the change smaller and let
"`parameters.xml` diff is empty" serve as a real regression gate.

### 3.7 Controller selection and startup

`MC_CTRL_ALG` selects the control law:

| value | controller |
|---|---|
| 0 | **default** — stock chain; `mc_controller` is not started at all |
| 1 | `CascadedPidController` — the stock cascade on this interface; the A/B baseline |
| 2 | `TemplateController` — skeleton to copy |

Selection is a plain switch in `controllers/ControllerRegistry.cpp` rather than a
code-generated registry. The `flight_mode_manager` generator exists because ~15
flight tasks need per-task Kconfig gating and a compile-time-sized union; none of
that applies here. A switch is greppable, gives a `-Wswitch` warning when an
enumerator is added without a case, and needs no build-system change.

`ROMFS/px4fmu_common/init.d/rc.mc_apps` branches:

```sh
control_allocator start

if param greater -s MC_CTRL_ALG 0
then
	mc_hover_thrust_estimator start
	flight_mode_manager start
	mc_controller start
else
	mc_rate_control start
	mc_att_control start
	mc_hover_thrust_estimator start
	flight_mode_manager start
	mc_pos_control start
fi
```

`param greater -s` is deliberate: `-s` makes a *missing* parameter fail silently,
so a build without `MODULES_MC_CONTROLLER` always takes the stock branch.

### 3.8 The output contract is singular

There is exactly one output contract: normalized torque and thrust, which
`control_allocator` mixes — you get `CA_*` geometry, desaturation,
`CA_FAILURE_MODE`, slew limits and reversible-motor handling for free.

An earlier revision let a controller own its mixing and publish `actuator_motors`
itself, gated by a `CA_EXT_ALLOC` parameter that silenced `control_allocator`
while armed. That was removed, for three reasons:

1. **It cannot be made sound with a static parameter.** The fallback swaps a
   direct-motors controller for the torque-output reference cascade *mid-flight*,
   so the module's output domain changes while the parameter still claims the old
   one. The result is that nobody publishes `actuator_motors` and the ESCs hold
   their last commanded value — `FunctionMotors` has no staleness check.
2. **The gate was too coarse.** `_publish_controls` guards all of
   `publish_actuator_controls()`, so it silenced `actuator_servos`, the
   motor-failure masks, `reversible_flags` and ICE shedding along with the motors.
3. **Yielding is not something `control_allocator` can express.** It publishes
   unconditionally on every `Run()` and is backup-scheduled every 50 ms, so a
   second publisher does not take ownership of the topic — it contends with it.

Failure-tolerant allocation therefore belongs behind the effectiveness matrix
(`CA_METHOD`, `CA_FAILURE_MODE`), which is `control_allocator`'s own extension
point, rather than behind a second publisher of the same topic.

### 3.9 Diagnostics

A new uORB message, `McControllerStatus`, carries the active algorithm, control
level, fallback state and reason, the published torque and thrust, per-stage
`dt`, update and invalid-output counters, and a free-form `debug[8]`
array a controller fills from `fillStatus()`.

Enum values are bound to the message constants with `static_assert`, so reordering
the C++ enum or editing the `.msg` breaks the build rather than silently
corrupting logs.

Registered with the logger as `add_optional_topic("mc_controller_status", 20)` —
"optional" because the topic is never advertised in stock builds and the logger
must not warn about it.

---

## 4. File inventory

### New

| path | purpose |
|---|---|
| `src/lib/mc_manual_mapping/StickToAttitudeSetpoint.{hpp,cpp}` | stick → attitude setpoint, extracted from `mc_att_control` |
| `src/lib/mc_manual_mapping/StickToRateSetpoint.{hpp,cpp}` | stick → body rates (Acro), extracted from `mc_rate_control` |
| `src/lib/mc_manual_mapping/StickToAttitudeSetpointTest.cpp` | differential test vs pre-extraction code |
| `src/lib/mc_manual_mapping/StickToRateSetpointTest.cpp` | differential test vs pre-extraction code |
| `src/lib/mc_manual_mapping/CMakeLists.txt` | library + test registration |
| `src/lib/mc_manual_mapping/SMOKE_LOG.md` | **the complete verification record** |
| `msg/McControllerStatus.msg` | framework status/diagnostics topic |
| `src/modules/mc_controller/ControllerApi/ControllerIO.{hpp,cpp}` | frozen input/output types |
| `src/modules/mc_controller/ControllerApi/MulticopterControllerBase.hpp` | the abstract controller class |
| `src/modules/mc_controller/ControllerApi/ControllerApiTest.cpp` | interface-contract tests |
| `src/modules/mc_controller/VehicleState/VehicleStateProvider.{hpp,cpp}` | uORB → `ControllerState`, filter chain, EKF resets |
| `src/modules/mc_controller/VehicleState/VehicleStateProviderTest.cpp` | filter/reset behaviour tests |
| `src/modules/mc_controller/CommandFrontEnd/ControlLevelResolver.{hpp,cpp}` | level resolution |
| `src/modules/mc_controller/CommandFrontEnd/ControlLevelResolverTest.cpp` | nav_state truth table vs commander's own table |
| `src/modules/mc_controller/CommandFrontEnd/CommandFrontEnd.{hpp,cpp}` | setpoint sourcing, takeoff ramp, failsafe, limits |
| `src/modules/mc_controller/CommandFrontEnd/CommandFrontEndTest.cpp` | setpoint-sourcing semantics |
| `src/modules/mc_controller/controllers/CascadedPidController.{hpp,cpp}` | stock cascade on the interface (baseline) |
| `src/modules/mc_controller/controllers/CascadedPidControllerTest.cpp` | differential test vs hand-wired stock triple |
| `src/modules/mc_controller/controllers/TemplateController.{hpp,cpp}` | skeleton controller |
| `src/modules/mc_controller/controllers/template_controller_params.c` | `MC_TPL_*` parameters |
| `src/modules/mc_controller/controllers/ControllerRegistry.{hpp,cpp}` | `MC_CTRL_ALG` → controller |
| `src/modules/mc_controller/MulticopterController.{hpp,cpp}` | the module + inner work item: uORB wiring, gyro-rate pipeline, output stage |
| `src/modules/mc_controller/OuterLoop.{hpp,cpp}` | optional trajectory work item on `nav_and_controllers` (Stage 12); active only when the controller reports `hasOuterStage()` |
| `src/modules/mc_controller/module.yaml` | `MC_CTRL_ALG`, `MC_CTRL_WD_MS` |
| `src/modules/mc_controller/Kconfig` | build option |
| `src/modules/mc_controller/CMakeLists.txt` (+ 4 sub-directory CMakeLists) | build |
| `src/modules/mc_controller/README.md` | usage guide |
| `src/modules/mc_controller/ARCHITECTURE.md` | this document |

### Modified

| path | change |
|---|---|
| `src/modules/mc_att_control/mc_att_control.{hpp,cpp}` | stick mapping extracted out; calls `StickToAttitudeSetpoint` |
| `src/modules/mc_rate_control/MulticopterRateControl.{hpp,cpp}` | Acro mapping extracted out; calls `StickToRateSetpoint` |
| `src/modules/mc_att_control/CMakeLists.txt`, `src/modules/mc_rate_control/CMakeLists.txt` | depend on `McManualMapping` |
| `src/lib/CMakeLists.txt` | add `mc_manual_mapping` |
| `msg/CMakeLists.txt` | register `McControllerStatus.msg` |
| `src/modules/logger/logged_topics.cpp` | log `mc_controller_status` |
| `ROMFS/px4fmu_common/init.d/rc.mc_apps` | branch on `MC_CTRL_ALG` |
| `boards/px4/sitl/default.px4board` | `CONFIG_MODULES_MC_CONTROLLER=y` |
| `boards/holybro/kakuteh7/default.px4board` | `CONFIG_MODULES_MC_CONTROLLER=y` |

The `Kconfig` `select`s `MODULES_MC_{POS,ATT,RATE}_CONTROL`. That is load-bearing,
not cosmetic: `PositionControl`, `AttitudeControl`, `Takeoff` and `GotoControl` are
libraries defined *inside* those module directories, and `MPC_*`/`MC_*` parameters
are only scanned when the owning module is enabled. Disabling them would delete
both the libraries and the parameters.

---

## 5. Stage-by-stage record

The work was built bottom-up. Each stage had to pass **BUILD** (both SITL and the
Kakute H7 flight board), **SMOKE** (does the new component do what it claims), and
**REGRESSION** (has anything that worked stopped working) before the next stage
started.

The full log, including every failure, is in
[`src/lib/mc_manual_mapping/SMOKE_LOG.md`](../../lib/mc_manual_mapping/SMOKE_LOG.md).

### Stage 0 — Capture a baseline (no code changes)

Everything later is compared against this, so it came first.

- `make px4_sitl_default` → **1122/1122 targets, exit 0**
- `make tests` → **154/154 passed, 0 failed.** Recording *which tests already
  fail* matters: without it, a pre-existing failure at Stage 4 looks like a new
  regression. There were none.
- `parameters.xml` saved as the reference for later diffs
- git SHA recorded: `154dc85809`

**Blocked:** SITL would not run. The machine had Gazebo Classic 11 + Ignition
Fortress, but this PX4 tree expects **Gazebo Harmonic**. CMake probed, found
`gz-sim_DIR: NOTFOUND`, and silently skipped the bridge — so only the `none`
simulator targets existed. Resolved later (Stage 1.5).

### Stage 1 — Extract the shared stick mapping

**Changes:** created `src/lib/mc_manual_mapping/` and refactored both stock modules
to use it (§3.6).

**Why this got its own stage:** it modifies the very code the reference controller
is supposed to be identical to. A regression here would silently redefine the
baseline, and every later comparison would be measured against a moved target.

**Tests — differential, not golden-value.** Each test file contains a **verbatim
copy of the pre-extraction implementation** (taken at `154dc85809`) and drives it
and the extracted class through an identical seeded-random sequence:

- `StickToAttitudeSetpointTest`: 2000 steps × 4 configurations (multicopter, VTOL
  tilt correction, `MPC_THR_CURVE` 1 and 2), randomized sticks/attitudes/`dt`,
  periodic hover-thrust updates including the invalid-estimate path, EKF heading
  resets, mode-exit resets. Compares `q_d[4]`, `thrust_body[3]`,
  `yaw_sp_move_rate` with `ASSERT_FLOAT_EQ`.
- `StickToRateSetpointTest`: 5000 randomized steps, plus targeted checks that
  centred sticks give zero rates and full stick reaches exactly
  `MC_ACRO_{R,P,Y}_MAX`.

**Results:** builds green on both targets; `parameters.xml` diff **empty**; 2/2
tests pass; full suite 156/156 (154 + 2 new).

**Also flown:** pristine stock (in an isolated git worktree at `154dc85809`) vs the
post-extraction tree, to catch a *wiring* error the unit tests cannot see — they
test the library in isolation, not its integration into `mc_att_control`.

**A methodology error found here, worth reading.** The first comparison used
whole-log RMS and reported **10 signals failing at 15–44%** — apparently a serious
regression. It was an artifact: the two runs idled on the ground for different
durations (37.7 s vs 6.8 s to reach altitude), so each log contained a different
*proportion* of ground/climb/cruise, and whole-log RMS measured that rather than
the control code. Corrected by windowing each log to its in-air segment,
re-zeroing time, truncating to the common overlap, and measuring position accuracy
as tracking error against the vehicle's own commanded setpoint. After correction,
7 of 8 signals agreed within 1%.

### Stage 1.5 — Install Gazebo Harmonic

Gazebo Harmonic 8.14.0 installed; Gazebo Classic removed (both claim
`/usr/bin/gz`). The build directory had to be **deleted**, not rebuilt: the
`NOTFOUND` result was *cached* in `CMakeCache.txt`, so an incremental build would
have kept reporting no simulator.

`parameters.xml` grew 16559 → 19306 lines because the gz bridge compiled for the
first time. Verified this was environmental, not code: **0 lines removed, and every
addition a `SIM_GZ_*` parameter.** Re-baselined for later stages.

### Stage 2 — Status message and logging

**Changes:** `msg/McControllerStatus.msg`, registered in `msg/CMakeLists.txt`, and
`add_optional_topic("mc_controller_status", 20)` in the logger.

Beyond the field list, named constants (`CONTROL_LEVEL_*`, `FALLBACK_*`) are declared in the message rather than left as magic numbers, so the
enum values live in one place.

**Results:** both builds green, header generated, **0 parameters** added by this
stage, suite 156/156.

### Stage 3 — Freeze the interface

**Changes:** `ControllerApi/` — the types and abstract class of §3.3. The module's
`CMakeLists.txt` deliberately did *not* yet call `px4_add_module()`, so nothing
linked into the flight image.

**Tests:** fail-closed defaults; `reset()` restoring them; finiteness checks;
thrust being mandatory alongside torque; level-bit composition (including that
`None` is never supportable); `supportsLevel()` masking for a partial-stack
controller; the `reset_integrals` contract being reachable; and `sizeof` budgets
for the structs copied every gyro cycle. (Originally 13 cases; the motor-channel
and output-type cases went with the direct-motors path — see Stage 10.)

**A failure and a wrong diagnosis.** The test initially failed to *link*
(`orb_publish`, `uorb_start`, `init_app_map` undefined) because it used
`px4_add_functional_gtest`. Switching to `px4_add_unit_gtest` fixed it. I recorded
the cause as "the functional variant does not link from a module subdirectory" —
**which Stage 4 disproved**, since `VehicleState/` uses the functional variant from
a sibling module subdirectory and links fine. The log was corrected. The rule that
actually holds: use the unit variant when the code needs nothing from the runtime,
the functional variant when it uses `DEFINE_PARAMETERS`.

**Results:** both builds green, params diff empty, suite 157/157.

### Stage 4 — `VehicleStateProvider`

**Changes:** port of `MulticopterPositionControl::set_vehicle_states()` — the
velocity notch → low-pass → derivative chain, position/velocity NaN semantics — plus
the EKF-reset bookkeeping from `adjustSetpointForEKFResets()`.

Three design decisions:

- **It owns no uORB subscriptions.** The module feeds it messages, which makes it
  unit-testable without the uORB runtime and lets tests drive synthetic EKF resets.
- **The first sample is never treated as a reset.** A fresh subscription arriving
  with non-zero reset counters would otherwise inject a spurious jump at every
  startup.
- **`endCycle()` is explicit, not folded into `getState()`.** `getState()` returns
  a reference so it cannot clear afterwards, and clearing at cycle start would wipe
  resets latched earlier in the same cycle.

**Tests (12 cases):** notch attenuation and DC pass-through; filter reset on
validity loss with a bounded derivative on regain; first-sample-is-not-a-reset;
reset deltas latched once then cleared; velocity-reset filter carry-over producing
no derivative spike; quaternion reset latching; freshness flags with position at
1/10 gyro rate; dt clamping on large gaps; hover-thrust parameter fallback.

**Explicitly not proven here:** numerical equality with stock's filter chain. That
arrives transitively at Stage 6.

**Results:** both builds green, params diff empty, suite 158/158.

**A harness bug:** the gate script reported failure while the suite passed — caused
by `make tests TESTFILTER="A|B"`. The Makefile expands `-DTESTFILTER=$(TESTFILTER)`
*unquoted*, so the shell treated `|` as a pipe and clobbered the test build
directory. Regex alternation cannot be passed through `TESTFILTER`.

### Stage 5a — Level resolution

Stage 5 was split: 5a is the correctness-critical level decision, 5b the large
mechanical port. Splitting kept each gate meaningful.

**Changes:** `ControlLevelResolver` (§3.2).

**Why the test is trustworthy:** it **links commander's own `mode_util` library**
and calls `mode_util::getVehicleControlMode()` for every `nav_state`, then applies
Commander's own `flag_multicopter_position_control_enabled` derivation. It cannot
drift from commander, and it fails loudly if upstream adds a nav_state. It also
asserts the stick-gate equivalences of §3.2 per-`nav_state` rather than in prose.

**Mutation testing.** A test that passes first try proves nothing unless it can
fail. Two deliberate mutations were injected:

| mutation | result |
|---|---|
| drop the `flag_control_termination_enabled` guard | `TerminationAlwaysYieldsNone` **FAILED** |
| resolve `Attitude` before `Trajectory` | **3 tests FAILED** including the truth table |

Both caught; resolver restored and re-verified.

**Coverage:** 18-row nav_state truth table; all six OFFBOARD submodes (notably
`thrust_and_torque` → `None`, since it enables allocation without rates);
termination and rates-disabled → `None`; fixed wing never reaching `Trajectory`;
VTOL transition gating (a tailsitter mid-transition keeps the attitude loop, a
non-tailsitter does not).

**Results:** both builds green, params diff empty, suite 159/159.

### Stage 5b — Setpoint sourcing

**Changes:** `CommandFrontEnd` — the largest port. Trajectory front-end from
`MulticopterPositionControl::Run()` (failsafe, takeoff ramp, EKF-reset
application, constraints and offboard `want_takeoff`, limit envelope, on-ground
override), plus Attitude and BodyRate sourcing wired to the Stage 1 mappings.

**Weaker evidence than other stages, stated up front.** There is **no differential
oracle** here: the stock logic is embedded in a module `Run()` loop, not an
extractable function, so it cannot be compiled as a verbatim twin. These tests
assert documented stock *semantics*, not bit-equality. Real equivalence arrived at
Stage 8.

**Tests (14 cases):** NaN-per-axis pass-through; the 200 ms last-valid fallback
boundary and failsafe beyond it; both degraded-failsafe branches (blind-land when
horizontal velocity is unavailable, blind-descent at 0.3 m/s² when vertical is);
EKF reset applied exactly once; on-ground override with `reset_integrals`;
takeoff-ramp monotonicity; tilt limit slewing; Stabilized generating **and
publishing** an attitude setpoint; offboard attitude **consumed but not
republished** (the self-feedback path the design forbids); Acro rate generation;
offboard NaN rate axes falling back to measured rate; the `reset_integrals`
contract across level change / disarm / landing; termination producing no command.

**Two informative test failures:**

1. Five Trajectory tests returned NaN setpoints. **The test premise was wrong, not
   the code** — the on-ground override was correctly active. Until
   `TakeoffState::flight`, the front end substitutes an empty setpoint plus
   100 m/s² downward acceleration so no thrust is produced and integrators cannot
   wind up pre-takeoff. Added a `takeOff()` helper that drives the state machine
   through spool-up and ramp as the real vehicle does.
2. Both failsafe tests. After takeoff the setpoint is fresh, so failsafe
   legitimately never triggers. Reaching it requires leaving position control,
   re-entering with a stale setpoint, **and** being past the 200 ms window.

**Results:** both builds green, params diff empty, suite 160/160.

### Stage 6 — The reference controller and the differential test

**Changes:** `CascadedPidController` — wraps the very same `PositionControl`,
`AttitudeControl` and `RateControl` objects the stock modules use, reusing their
parameters. `update()` dispatches on `command.level` and gates each stage on
`state.freshness` so the three stages keep their stock rates.

**The differential test.** A hand-wired stock triple is instantiated alongside the
controller and both are driven from an identical scripted sequence, requiring
agreement to **1e-6** on torque and thrust at every step:

- 3000 steps, mode sweep `Trajectory → Attitude → BodyRate → Trajectory`
- position samples at 1/10 gyro rate (exercises stage-rate parity)
- randomized attitude/velocity/rate wander so all three loops see real errors
- integral resets at each mode boundary
- an allocator saturation event mid-sequence

**Result: pass.** This is what made `MC_CTRL_ALG=1` numerically trustworthy before
anything flew, and it retroactively strengthened Stage 4 — the state provider's
filter chain feeds the position stage, so a numerical divergence there would
surface here.

Preserve-checklist items confirmed: yaw output LPF (`MC_YAW_TQ_CUTOFF`, with a
dedicated test driving a Nyquist-rate square wave), autotune additive rate
injection, integral reset on disarm, allocator anti-windup, `resetIntegralXY()`
when horizontal is uncontrolled.

**Three defects fixed:** a `MPC_*/MC_*` string inside a block comment (the `*/`
terminated it early); missing `[[fallthrough]]` markers under
`-Werror=implicit-fallthrough`; and an integrator assertion that **encoded a false
expectation** — `reset_integrals` zeroes the integrator and then the rate stage
runs in the same cycle, legitimately re-accumulating one step, exactly as stock
does. Asserting exact zero was wrong; it now asserts the wound-up history is gone.

**Results:** both builds green, params diff empty, suite 161/161.

### Stage 7 — The module and the output stage

**Changes:** `MulticopterController` (the `WorkItem`, uORB wiring, output stage),
`ControllerRegistry`, `TemplateController`, `module.yaml`,
`template_controller_params.c`.

**Live pipeline, disarmed SITL, stock modules stopped first** (two publishers on
`vehicle_torque_setpoint` would be meaningless data, not a test):

```
vehicle_torque_setpoint   250 Hz   (== vehicle_angular_velocity 250 Hz)
vehicle_thrust_setpoint   250 Hz
mc_controller_status      update_count climbing, invalid_output_count 0,
                          fallback_active False, control_level 3 (Trajectory)
                          dt 0.004, dt_attitude 0.004, dt_position 0.008
```

`dt_position` correctly decoupled from `dt` — stage-rate parity holding in the real
module, not only in tests.

**Safety-net verification, the point of this stage.** `TemplateController` was
temporarily patched to emit `torque(0) = NAN`:

| observation | result |
|---|---|
| `fallback_active` | True |
| `fallback_reason` | 2 = `INVALID_OUTPUT` |
| `mavlink_log_critical` | fired |
| published torque | finite, no NaN |
| status | `fallback : LATCHED (reason 2)` |

**A bug found by chasing an inconsistent counter.** The run passed every stated
assertion, but reported `invalid_output_count: 1001` *equal to*
`update_count: 1001`. By the control flow those must diverge once the latch sets,
so the anomaly was investigated rather than accepted. The disarm-clear was
**level-triggered**:

```cpp
if (!_vehicle_control_mode.flag_armed && _fallback_latched) { _fallback_latched = false; ... }
```

Disarmed on the bench that clears the latch every cycle, re-runs the bad
controller, re-latches, and logs again — ~250 log lines per second, and the
documented "sticky for the armed period" property silently did not hold. Fixed to
be **edge-triggered** on the armed→disarmed transition. After the fix:
`invalid_output_count: 1`.

The *protection* was never broken — NaN was always caught and the reference always
produced valid finite output. The defect was in the latch's persistence and
logging.

**Other defects:** `actuator_controls_status_s` has no `timestamp_sample` member;
and a **CMake ordering trap** — `px4_add_module()` calls `get_target_property()` at
*configure* time, and `src/modules/mc_controller` is processed before
`mc_pos_control` defines `PositionControl`/`Takeoff`, so those cannot appear in
the module's `DEPENDS`. They arrive transitively via the sub-libraries' `PUBLIC`
linkage, which resolves at generate time.

**A NuttX-only build break, caught because both targets are gated:**

```
error: format '%u' expects argument of type 'unsigned int',
       but argument 4 has type 'uint32_t' {aka 'long unsigned int'}
```

`uint32_t` is `unsigned int` on x86_64 but `long unsigned int` on 32-bit ARM, so
`PX4_INFO("%u", _update_count)` compiled cleanly for SITL and failed on the Kakute
H7. A SITL-only workflow would have shipped a broken flight-board build.

**Results:** both builds green, suite 161/161, style clean, `MC_CTRL_ALG` default 0.

### Stage 8 — Startup wiring, first flight, statistical A/B

**Changes:** `rc.mc_apps` branch (§3.7); board configs enabled.

**Boot directions:** with `MC_CTRL_ALG=1`, `mc_controller` runs and all three
stock modules are absent. (The `=0` assertion did not execute directly in that run
because the parameter had been left at 2 by the Stage 7 test, though the stock
branch is exercised by all four stock A/B flights.)

**Two real defects found by flying against actual stock:**

**Defect 1 — intermediate topics were not published.** `vehicle_rates_setpoint`
and `vehicle_attitude_setpoint` were **entirely absent** from the framework log
(stock had 3745 and 1499 samples). At Trajectory level the code only published
what the front end generated from sticks, so nothing was published when the
*controller* generated them. Consumers: WeatherVane, mavlink `ATTITUDE_TARGET`,
gimbal stream, QGC. Fixed, publishing only where the framework generated the value
and never where it consumed one.

**Defect 2 — trajectory yaw feed-forward was being dropped.** At Trajectory level
the attitude stage was called with `yaw_sp_move_rate = 0.f`. Stock forwards
`vehicle_attitude_setpoint.yaw_sp_move_rate`, which `PositionControl.cpp:272` sets
from the trajectory's commanded yaw rate. The framework silently lost trajectory
yaw feed-forward, so yaw lagged on every commanded heading change.

**Stage 6's differential test passed straight through this**, because the
hand-wired twin in that test contained the same mistake. **A reference twin
written by the same author reproduces the author's misunderstanding**; 1e-6
agreement only proved two copies of one wrong idea agreed. This is a structural
limit of differential testing against a self-written twin, and precisely why the
SITL A/B against real stock was not redundant.

Effect of the fix:

| signal | before | after |
|---|---|---|
| torque roll | 10.9% | 2.8% |
| torque yaw | 9.1% | 1.6% |
| rate sp roll | 7.7% | 1.8% |
| rate sp pitch | 5.0% | 1.2% |
| rate sp yaw | 8.2% | 1.9% |

**Getting the acceptance criterion right.** The plan specified a flat "within 5%
RMS". That is unachievable for low-amplitude signals: pitch torque RMS is ~0.01
normalized, and its measured run-to-run noise came out at 1.9%, 3.0%, 4.3% and
7.4% on different occasions — so the derived tolerance swung between 3.8% and
14.8%, and the same result read as pass or fail depending on which pair was flown.

So the noise floor was characterised properly: **4 stock flights → 6 pairwise
comparisons**. Stock-vs-stock pitch torque varies by up to **17.1%**.

Then, because a single framework flight cannot distinguish a systematic offset
from an outlier, **3 framework flights** were flown too, and the question posed
correctly — *does the framework sit further from stock than stock sits from
itself?*

```
signal            within-stock      within-fw          between      verdict
                    max (mean)     max (mean)       max (mean)
torque roll        5.7 ( 2.9)     9.0 ( 6.1)      9.9 ( 4.1)  noise-dominated
torque pitch      17.1 ( 9.3)    12.2 ( 8.2)     18.0 ( 6.9)  noise-dominated
torque yaw         3.7 ( 2.1)     1.4 ( 1.0)      3.5 ( 1.6)  noise-dominated
thrust z           0.1 ( 0.0)     0.1 ( 0.0)      0.1 ( 0.0)  noise-dominated
rate sp roll       5.1 ( 2.7)     3.5 ( 2.3)      4.8 ( 2.0)  noise-dominated
rate sp pitch      2.8 ( 1.4)     0.8 ( 0.5)      2.0 ( 0.9)  noise-dominated
rate sp yaw        3.6 ( 2.0)     1.2 ( 0.8)      2.9 ( 1.4)  noise-dominated

NO SIGNAL DIFFERS BEYOND RUN-TO-RUN NOISE
```

The decisive figure is roll torque: **within-framework spread (9.0% max) exceeds
the between-group spread (9.9% max, 4.1% mean)** — the framework differs from
itself as much as from stock. An intermediate verdict of "roll DIFF" was an
artifact of judging from one flight.

Position tracking error vs commanded setpoint: within 1.6–2.7% on all axes.

**Framework health across a full armed flight:**
`samples=6160, fallback_active_ever=no, invalid_output_count_max=0` — which closed
the Stage 7 gap about fallback stickiness across an armed period.

**A false claim corrected.** The Stabilized segment was reported as closing the
Stage 1 coverage gap on the strength of the script printing `stick sweep
complete`. The log showed `nav_state 15` (STAB) never occurred and `control_level`
was Trajectory for every sample: the mode switch was ACKed but never took, because
PX4 requires a live manual-control stream *before* accepting a manual mode. **A
COMMAND_ACK is not proof a mode engaged.** Fixed by streaming `MANUAL_CONTROL`
first and verifying engagement via HEARTBEAT. Verification flight:

```
nav_states:    [2, 4, 5, 14, 15, 18]        <- 15 = STAB present
control_level: {Attitude: 656, Trajectory: 5681}
fallback ever: False   invalid_max: 0
```

### Stage 9 — Template controller and documentation

**Changes:** `README.md`; `TemplateController` verified.

| gate | result |
|---|---|
| `MC_CTRL_ALG=2` flies the full profile | 31874 updates, 0 invalid, no fallback |
| unsupported-level fallback | see below |

The second fallback trigger, forced by removing `Trajectory` from
`supportedLevels()`:

```
WARN [mc_controller] template does not support all control levels (0x06);
                     the reference cascade will take over for the rest
control_level: 3   fallback_active: True   fallback_reason: 1  (UNSUPPORTED_LEVEL)
```

This is the trigger that matters for a rate-only fault-tolerant allocator.

### Stage 10 — `CA_EXT_ALLOC` and the direct-motors path — **SUPERSEDED**

> Everything in this stage was subsequently reverted. The direct-motors output
> path, `CA_EXT_ALLOC`, `MC_CTRL_EXT_ALC` and `OutputType` are all gone; the
> `control_allocator` change was backed out and that module is stock again. See
> §3.8 for why a static parameter cannot make a second `actuator_motors` publisher
> safe. The results below stand as a record of what was measured at the time —
> note in particular that the auto-land gate passing did **not** exercise the
> fallback, which is the case that actually breaks.

**Changes:** `CA_EXT_ALLOC` in `control_allocator` (§3.8);
`DirectMotorController` registered as `MC_CTRL_ALG=3`.

| gate | result |
|---|---|
| controller identity | `direct_motors`, `output type: actuator_motors` |
| flew the full profile | 33616 updates, 0 invalid, no fallback |
| **auto-land completed and DISARMED** | **pass** — proves `vehicle_thrust_setpoint` still feeds `land_detector` through the bypass |
| `actuator_motors` values | 1297/1297 finite, range [0.000, 1.000] |
| `actuator_motors` at gyro rate | **PARTIAL** — the 10 Hz in the log is the *logger's* per-topic downsample, not the publication rate; `uorb top` was sampled while disarmed |

A `CA_EXT_ALLOC` insertion into `control_allocator/module.yaml` initially broke the
YAML parse because that file uses 4-space nesting (parameters at 8 spaces) rather
than the 2-space style assumed. Restored from git and re-inserted programmatically
at the correct depth.

### Stage 11 — Fault-injection equivalence — **NOT VERIFIED**

**Goal:** confirm the framework fails the same way as stock under a rotor loss, so
`MC_CTRL_ALG=0` vs `=1` is a valid control pair when evaluating a fault-tolerant
controller.

**Outcome: no rotor was ever killed. The stage tested nothing.**

First attempt: both arms flew, both printed `PROFILE COMPLETE`, both logs
captured, exit 0. Buried in the output: `ERROR [failure] Failure type '1' not
found`. The syntax is `failure <unit> <type> -i <instance>`; the instance had been
put where the type belongs.

Second attempt with corrected syntax reported no error, but ULog analysis shows no
failure in either arm:

```
stock:      motor_failure_mask max=0, nonzero_samples=0, all 4 motors reach 1.000
framework:  motor_failure_mask max=0, nonzero_samples=0, all 4 motors reach 1.000
```

`FailureInjector` never logged `CMD_INJECT_FAILURE`, so it never processed the
command — most plausibly because `SYS_FAILURE_EN` was set at runtime and
`_failure_injection_enabled` had not picked it up.

**What can and cannot be claimed.** It cannot be claimed that the framework and
stock respond identically to a rotor loss. The available *argument* — not a
measurement — is that the branch's `kill_switch_2` injector lives in
`mixer_module`, **downstream of `actuator_motors`**, so it is architecturally
independent of which controller produced the moments, and Stage 8 established that
the upstream torque/thrust are statistically indistinguishable between arms.

**How to close it:** set `SYS_FAILURE_EN=1`, `param save`, **reboot**, then inject.
Or, preferred here, verify with the branch's own `kill_switch_2` on hardware with
the RC transmitter — that path is the one that actually flies, and it cannot be
driven over MAVLink because it reads `manual_control_switches` from real RC.

### Stage 12 — Stock work-queue separation

Until this stage everything ran on `rate_ctrl`, including `PositionControl`. Stock
keeps that math on `nav_and_controllers` for a reason: the gyro loop has a 2.5 ms
budget on an H7 at `IMU_GYRO_RATEMAX=400`.

The constraint was that a **monolithic** full-stack law — the framework's whole
purpose — cannot be split across two threads. So the separation is opt-in through
three additive virtuals that default to today's behaviour, `hasOuterStage()`
foremost. False → `OuterLoop::Run()` returns immediately and the entire law runs at
gyro rate, unchanged.

Result:

```
STOCK   wq:rate_ctrl            mc_rate_control       250.0 Hz
        wq:nav_and_controllers  mc_att_control        250.0 Hz
                                mc_pos_control        125.0 Hz
SPLIT   wq:rate_ctrl            mc_controller         250.0 Hz
        wq:nav_and_controllers  mc_controller_outer   125.0 Hz
```

Gates: both builds clean, **161/161** tests, 3 clean split flights, measured rates
inner 250.0 Hz / attitude 247.6–249.6 Hz / position 125.0 Hz, inner:outer update
ratio 2.24 ≈ 250/125.

**Two defects found.**

The first stopped the vehicle leaving the ground. `timestamp_sample` is written by
`updateAngularVelocity()`, which a position-driven work item never calls, so the
outer loop passed `now = 0`; `TakeoffHandling` read that as "now", never left
rampup, and thrust stayed pinned at −0.001. No earlier gate could have caught it —
every unit test drives the state provider through the gyro path, because until this
stage every consumer did.

The second is the more instructive one, and is covered in §7 below because the
lesson is about the verification method rather than the code.

### A pattern across four stages

Four times a green result was not evidence:

| stage | reported | actually |
|---|---|---|
| 7 | every NaN assertion passed | latch re-triggering every cycle, logging at 250 Hz |
| 8 | script printed `stick sweep complete` | STABILIZED never engaged (nav_state 15 absent) |
| 11 | both profiles completed, logs captured | injection rejected, then silently not applied |
| 12 | A/B: every signal inside run-to-run noise | a topic published in the wrong flight mode, at gyro rate |

**A script's success message is not evidence the thing under test happened.** Every
SITL gate needs an independent check in the ULog or a status topic. This
generalises directly to evaluating a fault-tolerant controller: verify from the log
that the failure actually occurred before believing any comparison.

Stage 12 extends the pattern from scripts to *metrics*: a passing statistical
comparison is not evidence either, when it measures the wrong quantity.

---

## 6. Test inventory

Baseline was 154 test binaries; the work adds 7, all passing (161/161).

| test | type | what it proves | what it does **not** prove |
|---|---|---|---|
| `functional-StickToAttitudeSetpoint` | differential vs verbatim pre-extraction code | the stick→attitude mapping is unchanged by extraction, bit-for-bit over 2000 steps × 4 configs | that it is wired into `mc_att_control` correctly (Stage 1 flight covers that) |
| `functional-StickToRateSetpoint` | differential vs verbatim pre-extraction code | the Acro mapping is unchanged, 5000 steps | in-situ Acro behaviour (never flown) |
| `unit-ControllerApi` | contract | interface is implementable; defaults fail closed; NaN/motor validation; level-mask composition | any behaviour — these are type definitions |
| `functional-VehicleStateProvider` | behavioural | filter DC pass-through and notch attenuation, reset semantics, dt clamps, freshness flags | numerical equality with stock's filter chain (Stage 6 covers it) |
| `unit-ControlLevelResolver` | truth table vs commander's own `mode_util` | level resolution matches stock's gating for every `nav_state`; mutation-tested to prove it can fail | nothing about setpoint content |
| `functional-CommandFrontEnd` | semantic | failsafe timing boundaries, EKF-reset-once, on-ground override, ramp monotonicity, publish/consume rules | bit-equality with stock — **no oracle exists** for this stage |
| `functional-CascadedPidController` | differential vs hand-wired stock triple @ 1e-6 | the reference controller reproduces the stock cascade numerically across a 3000-step mode sweep | **a shared misunderstanding of stock** (it passed through the yaw feed-forward defect), and inter-module latency differences |

Plus SITL gates, which are scripted rather than in CI:

| gate | stage | result |
|---|---|---|
| pipeline live, 250 Hz publication, disarmed | 7 | pass |
| forced-NaN → fallback latched, finite output only | 7 | pass |
| both `rc.mc_apps` boot directions | 8 | `=1` verified; `=0` indirectly |
| statistical A/B, 4 stock + 3 framework flights | 8 | no signal differs beyond noise |
| Attitude level executes in flight | 8 | pass (656 samples at Attitude) |
| framework health across armed flight | 8 | 0 fallbacks, 0 invalid outputs |
| template controller flies | 9 | pass |
| unsupported-level fallback | 9 | pass |
| direct-motors flies and auto-land disarms | 10 | superseded — path removed |
| fault-injection equivalence | 11 | **not verified** |

---

## 7. Known limitations and unverified items

### Structural limitations of the verification

- **The Stage 6 differential test compares against a self-written twin**, in one
  process with no uORB hop. It cannot detect a shared misunderstanding of stock
  (demonstrated: it passed the dropped yaw feed-forward), nor inter-module latency
  differences. The SITL A/B against real stock is not redundant.
- **Stage 5b has no differential oracle at all.** Its guarantees are semantic.
- **SITL is not run-to-run deterministic**, so all flight comparisons are
  statistical. Low-amplitude torque signals are noise-dominated in relative terms
  (stock-vs-stock pitch torque varies up to 17%).
- **Comparing signal values does not compare behaviour.** This one cost real
  debugging time at Stage 12 and is worth stating precisely.

  The Stage 8 and Stage 12 A/Bs compared *magnitudes* — torque, thrust, rate
  setpoints — and both passed. Neither could see that
  `vehicle_local_position_setpoint` was being published in flight modes where stock
  publishes nothing at all, because the numbers inside it looked entirely
  plausible: a frozen −4.97 m altitude setpoint is indistinguishable from a live
  one until you ask *when* it was published.

  What exposed it was measuring position tracking error against the vehicle's own
  commanded setpoint — z-RMS 1.497 m against stock's 0.120 m, reproducible to
  ±0.005 across three flights. Localising by thirds of the window put all of it in
  the Stabilized segment; counting samples per flight mode found stock publishing
  0 and the framework publishing 146.

  The vehicle was flying correctly the entire time. The real consequences are a
  stale reset origin handed to the flight tasks, and log analysis that lies.

  The gate that catches this class asks *which topics are published in which
  modes*: `pub_matrix.py` counts every intermediate topic per control-level segment
  and diffs stock against the framework. The plan called for exactly this assertion
  in §3 ("Publish/consume rule — assert in review"); it was applied to
  `vehicle_attitude_setpoint` and `vehicle_rates_setpoint` and missed for
  `vehicle_local_position_setpoint`, whose publication sat three lines below a
  correctly-gated one.

  **A flight comparison must check the publication matrix, not only the signal
  values.** Requires a Stabilized segment in the profile (`PROFILE_STAB=1`) — the
  defect lives in modes a pure-Offboard flight never enters, which is exactly why
  the Stage 8 framework flight did not reveal it.

- **A whole-flight aggregate can mask a large error confined to one flight mode.**
  The same A/B flagged `thrust z` at 1.9% — barely above the noise floor, and easy
  to dismiss alongside two genuinely marginal signals in the same table. Chasing it
  found the manual throttle curve running on the `MPC_THR_HOVER` parameter (0.60)
  instead of the live hover-thrust estimate (0.72): a **21% collective error in
  Stabilized**, averaged down to 1.9% because Stabilized is ~15 s of a 66 s window
  and the Trajectory segments were always correct.

  Three flights commanded a collective of exactly −0.6000, the parameter value, and
  stock commanded −0.7257. After the fix: −0.7258.

  Per-mode analysis is not optional in either direction — it catches both the small
  artifact inflated by a bad window (the phantom 12× above) and the large error
  diluted by a good one. `pub_matrix.py` segments by control level for this reason.

### Not verified

| item | why | how to close |
|---|---|---|
| **Real hardware** | nothing has run on the Kakute H7 beyond compiling | bench disarmed → `mc_controller status`, `work_queue status`, `perf` → tethered STAB hover → ALTCTL → POSCTL |
| **Fault-injection equivalence** | injection never applied (Stage 11) | `SYS_FAILURE_EN=1` → `param save` → reboot → inject; or test `kill_switch_2` with the transmitter |
| **Acro / BodyRate level in flight** | deliberately skipped — scripted acro has high crash risk and the mapping is proven bit-exact by unit test | fly it manually |
| **Armed `actuator_motors` publication rate** | measured 10 Hz was the logger's downsample | `uorb top` while armed |
| **`MC_CTRL_ALG=0` boot assertion** | parameter had been left at 2 in that run | boot with the default and check `mc_controller` is absent |
| **VTOL** | out of scope; the module refuses to start on a VTOL airframe | — |
| **Inner-loop cycle time** | **not measurable in SITL** — every `PC_ELAPSED` counter reports `0us elapsed`, for the stock modules too | hardware only: `perf` → `mc_controller: cycle` against the 2.5 ms budget at `IMU_GYRO_RATEMAX=400` |
| **The outer/inner disjoint-state contract** | enforced by review, not the compiler | audit any controller that sets `hasOuterStage()`; only `CascadedPidController` does today |
| **`rate sp pitch` / `rate sp yaw`** | flagged by the Stage 12 A/B, but marginal — the between-group max sits inside stock's own spread (16.6 vs 15.5); only the means differ | more flights, or accept as noise |

### Behavioural differences from stock that are real, not bugs

- **The attitude stage sits on `rate_ctrl`, not `nav_and_controllers`.** Stock runs
  the attitude loop on `nav_and_controllers` at 250 Hz; here it runs beside the rate
  loop on `rate_ctrl`. Deliberate: the Brescianini quaternion P law is cheap, and
  co-locating it removes a uORB hop of latency from the fast cascade, while the
  expensive `PositionControl` math is what actually needed to move off the gyro
  path. Matching stock exactly would require a third work item (attitude at 250 Hz
  and position at 125 Hz sharing `nav_and_controllers`).
- **In monolithic mode the attitude→rate uORB hop is gone entirely.** In stock,
  `mc_att_control` publishes `vehicle_rates_setpoint` and `mc_rate_control` consumes
  it, costing roughly one cycle of latency; a monolithic controller runs both stages
  in the same cycle. Less lag is arguably better control, but it *is* different, and
  the Stage 6 differential test structurally cannot see it. The Stage 8 and Stage 12
  statistical A/Bs found no signal outside run-to-run noise, so any effect is below
  the measurement floor.

---

## 8. How to use it

### Enable the framework

```
param set MC_CTRL_ALG 1     # 1 = reference cascade, behaviourally equal to stock
reboot                      # 0 <-> non-zero changes which modules start
mc_controller status
```

Reverting is always `param set MC_CTRL_ALG 0` + reboot.

### Add your own controller

Five steps, detailed in [`README.md`](README.md):

1. copy `controllers/TemplateController.{hpp,cpp}`, rename the class
2. add `controllers/my_controller_params.c` with `MC_MY_*` parameters
3. add both files to `SRCS` in `controllers/CMakeLists.txt`
4. add an `Algorithm` enumerator and one `case` in `ControllerRegistry`
5. add the value to `MC_CTRL_ALG` in `module.yaml`

### Rules that will bite

- honour `command.reset_integrals`
- always fill `output.thrust`, not just `output.torque`
- `update()` is real-time: no allocation, no blocking, no `printf`
- declare every level you handle in `supportedLevels()`
- leave `hasOuterStage()` false unless your law genuinely separates — if you do opt
  in, `updateOuter()` and `update()` run concurrently on different work queues and
  must touch disjoint members

### Re-check equivalence after any change

```
make tests TESTFILTER=CascadedPid            # differential vs stock triple
make tests TESTFILTER=ControlLevelResolver
make tests TESTFILTER=CommandFrontEnd
make tests TESTFILTER=VehicleState
make tests TESTFILTER=ControllerApi
make tests                                    # full suite, expect 161/161
```

Unit tests cannot see cross-module or cross-queue behaviour. After any change that
touches publication, work-queue placement or the front end, re-fly the A/B — and run
**both** checks, because they catch different classes of defect:

```
# 1. signal magnitudes: is the control output equivalent?
python3 group_compare.py --stock <stock.ulg ...> --fw <framework.ulg ...>

# 2. publication matrix: are the right topics published in the right modes?
python3 pub_matrix.py <stock.ulg> <framework.ulg>
```

Both need ≥3 flights per arm for a valid within-group spread (n=1 collapses to
comparing against stock's spread alone, which produced a phantom regression at
Stage 8), and the profile must include a Stabilized segment (`PROFILE_STAB=1`) or
the modes where publication defects live are never entered. Check #2 is what found
the Stage 12 defect that #1 passed.

`TESTFILTER` cannot take regex alternation — run one filter per invocation.

### Diagnostics

```
mc_controller status            # controller, level, fallback state, counters
listener mc_controller_status   # torque/thrust/motors, per-stage dt, fallback reason
```
