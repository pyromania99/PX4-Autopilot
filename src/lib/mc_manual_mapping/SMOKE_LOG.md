# Pluggable MC controller framework — verification log

Durable record of every verification gate run, per the staged build plan.
Each stage records: what was built, the exact command, the observed result, and
what the gate explicitly does **not** prove.

Machine: Linux 6.8.0-136-generic, 16 cores.
Flight board target: `holybro_kakuteh7_default`.

> **Read this as a journal, not as current-state documentation.** Entries record
> what was true on the date at the top of each stage. One line of work was
> subsequently reverted in full: the direct-motors output path and its allocator
> gate — `OutputType`, `ControllerOutput::motors[]`, `DirectMotorController`,
> `MC_CTRL_ALG=3`, `MC_CTRL_EXT_ALC`, `CA_EXT_ALLOC` and the `control_allocator`
> change. Stages 3 and 10 describe them as built; Stage 10 carries the reversal
> and the reasoning. For current state see `mc_controller/ARCHITECTURE.md` §3.8.

---

## Stage 0 — Capture the baseline (no code changes)

**Date:** 2026-07-29
**Tree:** `154dc85809bbe3ba0f11612d1bdef49bf430fdc5` (branch `motor_failure`, clean)

| Gate | Command | Result |
|---|---|---|
| BUILD | `make px4_sitl_default` | **PASS** — exit 0, 1122/1122 targets |
| SMOKE | `make tests` | **PASS** — 154/154 passed, 0 failed, 55.14 s |
| ARTIFACT | `build/px4_sitl_default/parameters.xml` | saved as `baseline_parameters.xml`, 16559 lines |
| ARTIFACT | stock SITL flight log | **BLOCKED — not captured** |

**Known-failing tests at baseline: NONE.** Any test failure after this point is
introduced by this work.

### Blocker: SITL unavailable

`make px4_sitl gz_x500` fails with `ninja: error: unknown target 'gz_x500'`.
Root cause: this machine has Gazebo Classic 11 + Ignition Fortress
(`gz-transport11`, `ignition-common4`), but the tree expects **Gazebo Harmonic**
(`Tools/setup/ubuntu.sh:218` installs `gz-harmonic`). CMake probed and found
nothing — `gz-sim_DIR: NOTFOUND`, `gz-transport_DIR: NOTFOUND`,
`gazebo_DIR: NOTFOUND` — and silently skipped the bridge, so only the `none` and
`none_iris` simulator targets are registered. Gazebo Classic cannot substitute
either: only the `libgazebo11` runtime is installed, not `libgazebo11-dev`.

Resolving this needs `sudo apt install gz-harmonic libunwind-dev` (plus the OSRF
apt repo). Deferred by decision: Stages 1–7 are host-side and fully verifiable
without a simulator. **SITL is required at Stage 8 and Stage 10.**

**UNVERIFIED after Stage 0:** no runtime or flight behaviour baseline exists.
Only build health and unit-test health are established.

---

## Stage 1 — `src/lib/mc_manual_mapping/` extraction

**Date:** 2026-07-29

Extracted the manual stick mappings out of the stock modules into a shared
library so the reference controller and `mc_att_control`/`mc_rate_control` cannot
diverge:

- `StickToAttitudeSetpoint` ← `MulticopterAttitudeControl::generate_attitude_setpoint()`
  + `::throttle_curve()` + the slew-rate / EKF-reset / mode-exit-reset handling from `Run()`
- `StickToRateSetpoint` ← the Acro superexpo mapping in `MulticopterRateControl::Run()`

Net effect on the stock modules: **-205 lines, +19 lines** (pure relocation).

### Plan correction applied

The plan (§5) called for moving parameter *definitions* into the new library.
That was unnecessary and was **not** done. Definitions are already scattered
across the tree (`MC_AIRMODE` → `src/lib/mixer_module/params.c`,
`MPC_MAN_TILT_MAX` → `src/modules/mc_pos_control/multicopter_stabilized_mode_params.c`,
`MC_ACRO_*` → `src/modules/mc_rate_control/mc_acro_params.c`) and
`DEFINE_PARAMETERS` binds purely by name. Leaving every definition in place makes
the change smaller and keeps the parameters.xml gate meaningful.

### Gates

| Gate | Command | Result |
|---|---|---|
| BUILD (SITL) | `make px4_sitl_default` | **PASS** — exit 0 |
| BUILD (NuttX) | `make holybro_kakuteh7_default` | **PASS** — exit 0 |
| REGRESSION | `diff baseline_parameters.xml build/px4_sitl_default/parameters.xml` | **PASS — diff empty** |
| SMOKE | `make tests TESTFILTER=StickTo` | **PASS** — 2/2 |
| REGRESSION | `make tests` (full suite) | **PASS** — 156/156, 0 failed |
| STYLE | `Tools/astyle/check_code_style.sh` on all 8 touched files | **PASS** — no diagnostics |

```
1/2 Test #52: functional-StickToAttitudeSetpoint ...   Passed    0.01 sec
2/2 Test #53: functional-StickToRateSetpoint .......   Passed    0.00 sec
100% tests passed, 0 tests failed out of 2
```

Full suite: **156 = 154 (Stage 0 baseline) + 2 (new)**, 0 failed. The count
matching the baseline exactly is the regression evidence: no pre-existing test
changed state.

### How the smoke tests work

Both are **differential tests**, not golden-value tests. Each test file contains a
verbatim copy of the pre-extraction implementation (taken at `154dc85809`) and
drives it and the extracted class through an identical seeded-random input
sequence, requiring agreement at every step:

- `StickToAttitudeSetpointTest.cpp` — 2000 steps × 4 configurations (multicopter,
  VTOL tilt correction, `MPC_THR_CURVE` = 1 and 2), exercising randomized sticks,
  attitudes and `dt`, plus periodic hover-thrust updates (including
  invalid-estimate handling), EKF heading resets, and mode-exit resets. Compares
  `q_d[4]`, `thrust_body[3]` and `yaw_sp_move_rate` with `ASSERT_FLOAT_EQ`.
- `StickToRateSetpointTest.cpp` — 5000 randomized steps comparing rate and thrust
  setpoints, plus targeted checks that centred sticks give zero rates and full
  stick reaches exactly `MC_ACRO_{R,P,Y}_MAX`.

Plus two behavioural assertions on the extracted class: tilt magnitude clamps to
`MPC_MAN_TILT_MAX`, and the arming gesture (`throttle < -0.9` with
`MC_AIRMODE != 2`) unlocks the held yaw setpoint.

This is a deliberate improvement over the plan's original proposal of scraping
golden vectors from a SITL log: it is deterministic, needs no simulator, and
directly compares old code against new rather than against a recording.

### Issue found and fixed during this stage

First run of the smoke gate failed to compile (`TESTS_EXIT=2`):

```
error: 'virtual void StickToAttitudeSetpoint::updateParams()' is protected within this context
error: 'virtual void StickToRateSetpoint::updateParams()' is protected within this context
```

`ModuleParams::updateParams()` is protected and cascades to children
(`module_params.h:79-90`). The library classes were correct — `protected: void
updateParams() override` matches the in-tree convention (`GainCompression3d`,
`MixingOutput`) — but the tests called it directly. Fixed by parenting the objects
under a small `ParamHarness` that re-exposes `updateParams()`, which also
exercises the real parent→child cascade the owning module uses.

**UNVERIFIED after Stage 1:** no runtime behaviour was exercised. The extraction is
proven equivalent *as a function*, on host, over the sampled input space. It has
not been run on target hardware, in SITL, or in a real control loop. Flight
equivalence is not established until Stage 8.

**Disposition of these tests: PERMANENT.** They stay as regression guards so future
edits to the shared mapping cannot silently change stock flight behaviour.

---

## Stage 1.5 — Gazebo Harmonic installed (environment change)

**Date:** 2026-07-29

Gazebo Harmonic 8.14.0 installed via the OSRF repo; Gazebo Classic 11 was removed
(both packages claim `/usr/bin/gz`). `build/px4_sitl_default` was deleted and
re-configured from scratch — the `gz-sim_DIR: NOTFOUND` result was **cached** from
the original configure, so an incremental build would have kept reporting no
simulator even with Harmonic correctly installed.

Result: `gz_x500` and the rest of the `gz_*` simulator targets are now registered.
**Stage 8 and Stage 10 are unblocked.**

### Consequence: the Stage 0 parameters.xml baseline was invalidated

With the gz bridge compiling for the first time, `simulation/gz_bridge` now
contributes its parameters, so `parameters.xml` grew 16559 → 19306 lines.

Verified this is environmental and not caused by any code in this work:

```
REMOVED lines (<)                     : 0
ADDED params not matching ^SIM_GZ     : 0
```

i.e. **nothing was removed or altered; every addition is a `SIM_GZ_*` parameter.**

The pre-Gazebo file is kept as `baseline_parameters_pre_gz.xml`, and
`baseline_parameters.xml` has been **re-captured** so later stages compare against a
valid reference. Stage 1's empty-diff result remains valid: it was measured
before this environment change, against the matching baseline.

---

## Stage 2 — `McControllerStatus.msg` + logger registration

**Date:** 2026-07-29

- `msg/McControllerStatus.msg` — new non-versioned message. Named constants
  (`CONTROL_LEVEL_*`, `OUTPUT_TYPE_*`, `FALLBACK_*`) are declared in the message
  rather than left as magic numbers in comments, so the enum values live in one
  place and C++ can use `mc_controller_status_s::FALLBACK_INVALID_OUTPUT`.
- `msg/CMakeLists.txt` — registered alphabetically between `MavlinkTunnel.msg` and
  `MessageFormatRequest.msg`.
- `src/modules/logger/logged_topics.cpp` — `add_optional_topic("mc_controller_status", 20)`.
  `add_optional_topic` is correct because the topic is never advertised in stock
  builds, so the logger must not warn about it.

### Gates

| Gate | Command | Result |
|---|---|---|
| BUILD (SITL) | `make px4_sitl_default` (clean configure) | **PASS** — artifacts generated |
| BUILD (NuttX) | `make holybro_kakuteh7_default` | **PASS** — exit 0 |
| SMOKE | generated `uORB/topics/mc_controller_status.h` | **PASS** — 3950 bytes |
| REGRESSION | `parameters.xml` attributable to this stage | **PASS** — 0 params added/removed by this code |
| REGRESSION | `make tests` | **PASS** — 156/156, 0 failed |

**UNVERIFIED after Stage 2:** the topic has never been published or logged at
runtime — only generated and registered. The `listener mc_controller_status` /
ULog round-trip check needs a running SITL instance and is deferred to Stage 7,
when there is finally a publisher.

---

## Stage 0 (completed) — SITL flight baseline + Stage 1 flight-level regression

**Date:** 2026-07-29

### Flight profile

`flight_profile.py` (scratchpad): OFFBOARD climb to 5 m → 6 m square (4 legs,
8 s each) → yaw sweep 0/45/90/45/0° → AUTO.LAND → disarm. ~75 s, ~59 s in air.

Two environment findings, both now encoded in the script:

1. **Headless SITL will not arm** without something heartbeating as a GCS —
   arming returns `MAV_RESULT_TEMPORARILY_REJECTED` with
   `Preflight Fail: No connection to the GCS`. Fixed with a background thread
   sending `MAV_TYPE_GCS` heartbeats.
2. **`MAV_CMD_NAV_TAKEOFF` was ACKed but never climbed** — the vehicle entered
   AUTO.LOITER and then auto-disarmed on the ground. Replaced with an OFFBOARD
   position-setpoint climb, entering OFFBOARD *before* arming. This is also
   strictly better for A/B purposes: the trajectory is commanded, not delegated
   to the AUTO state machine, so it is repeatable between runs.

### Logs captured

| File | Tree | Purpose |
|---|---|---|
| `baseline_prestock.ulg` | pristine `154dc85809` (git worktree) | pre-extraction reference |
| `baseline_stock.ulg` | Stage 1+2 tree | post-extraction; **the Stage 8 A/B reference** |
| `baseline_stock_run2.ulg` | Stage 1+2 tree, repeat | noise-floor measurement |

### Methodology correction (important)

The first comparison used whole-log RMS and reported **10 signals failing at
15–44%**. That result was an artifact, not a regression: the two runs idled on
the ground for different durations (37.7 s vs 6.8 s to reach altitude), so each
log contained a different *proportion* of ground/climb/cruise, and whole-log RMS
measured that timing difference rather than the control code.

`compare_logs2.py` fixes this: window each log to its in-air segment
(alt ≥ 4 m), re-zero time, truncate both to the common overlap, and measure
position accuracy as tracking error against the vehicle's *own* commanded
setpoint (phase-independent).

### Result — Stage 1 flight-level regression gate

Phase-aligned over 59.1 s of level flight:

| signal | pre-extraction | post-extraction | rel | same-build noise floor |
|---|---|---|---|---|
| torque x (roll) | 0.0185 | 0.0186 | 0.6% | 4.0% |
| torque y (pitch) | 0.0133 | 0.0123 | 7.7% | **7.4%** |
| torque z (yaw) | 0.0677 | 0.0683 | 0.9% | 0.3% |
| thrust z | 0.7310 | 0.7310 | 0.0% | 0.0% |
| rates roll / pitch / yaw | — | — | 0.1 / 0.9 / 0.6% | 0.9 / 0.9 / 0.3% |
| pos err x / y / z | 1.070 / 1.057 / 0.124 m | 1.089 / 1.071 / 0.122 m | 1.8 / 1.3 / 1.2% | 1.4 / 0.4 / 0.4% |

**The pitch-torque outlier is SITL noise, not code.** The same build flown twice
differs by 7.4% on that signal — indistinguishable from the 7.7% measured between
pre- and post-extraction. Every pre-vs-post difference lies within the same-build
noise floor.

**VERDICT: Stage 1 flight-level regression gate PASSES.** The extraction is
equivalent both as a function (bit-exact unit tests) and in situ.

### Consequence for Stage 8's acceptance criteria

The plan's flat "within 5% RMS" threshold is **too tight for low-amplitude
signals**: `vehicle_torque_setpoint.xyz[1]` has a natural run-to-run spread of
~7.5% because its absolute magnitude (~0.012 normalized) makes relative
comparison noise-dominated. Stage 8 must use a per-signal tolerance of
`max(5%, 2 x measured noise floor)`, or compare absolute rather than relative
error for near-zero signals. Re-measure the noise floor at Stage 8 rather than
reusing these numbers.

### Coverage gap (open)

The profile is entirely OFFBOARD, i.e. Trajectory level. It exercises the
extracted code's *wiring* — `reset()`, `updateSlewRates()` and
`setHoverThrustEstimate()` all run every cycle in the non-manual branch — but it
never calls `StickToAttitudeSetpoint::update()` or `StickToRateSetpoint::update()`,
which only run in Stabilized and Acro. Those mappings are proven bit-exact by the
unit tests; what remains unexercised in situ is their call sites.

**To close at Stage 8:** add a Stabilized segment driven by `MANUAL_CONTROL`
(requires `COM_RC_IN_MODE` = 1 or 2). Scripted Acro is deliberately excluded —
high crash risk, negligible incremental value over the unit tests.

---

## Stage 3 — `ControllerApi/` interface freeze

**Date:** 2026-07-29

New `src/modules/mc_controller/` with the frozen interface only. The module's
`CMakeLists.txt` deliberately does **not** call `px4_add_module()` yet — that
arrives at Stage 7 — so this stage contributes a library and its tests without
linking anything into the flight image.

- `ControllerApi/ControllerIO.hpp` — `ControlLevel`, `OutputType`,
  `StateFreshness`, `EkfResets`, `ControllerState`, `ControllerCommand`,
  `ControllerOutput`, `AllocatorFeedback`, `outputIsFinite()`.
- `ControllerApi/MulticopterControllerBase.hpp` — the abstract class.
- `Kconfig` — `select`s `MODULES_MC_{POS,ATT,RATE}_CONTROL`. Load-bearing:
  `PositionControl`, `AttitudeControl`, `Takeoff` and `GotoControl` are libraries
  defined *inside* those module directories, and the `MPC_*`/`MC_*` parameters
  are only scanned when their owning module is enabled.
- Enabled `CONFIG_MODULES_MC_CONTROLLER=y` on `boards/px4/sitl/default.px4board`
  and `boards/holybro/kakuteh7/default.px4board`.

### Two decisions worth recording

**Defaults fail closed.** `ControllerOutput` initialises `valid = false` with
torque/thrust NaN, so a controller that forgets to write its output is rejected
by the output stage rather than silently commanding zero torque — which on a
multicopter is free fall. `ControllerState` defaults match that intent
(`landed = true`, `armed = false`, all validity flags false).

**The enums cannot drift from the message.** `ControlLevel` and `OutputType` are
published directly as `mc_controller_status` fields, so `ControllerIO.cpp` carries
`static_assert`s binding every enumerator to its `mc_controller_status_s::`
constant. Reordering the enum or editing the `.msg` breaks the build instead of
quietly corrupting the logs.

### Gates

| Gate | Command | Result |
|---|---|---|
| BUILD (SITL) | `make px4_sitl_default` | **PASS** — exit 0 |
| BUILD (NuttX) | `make holybro_kakuteh7_default` | **PASS** — exit 0 |
| REGRESSION | `parameters.xml` vs baseline | **PASS** — diff empty |
| SMOKE | `make tests TESTFILTER=ControllerApi` | **PASS** — `unit-ControllerApi` 1/1, 13 test cases |
| REGRESSION | `make tests` | **PASS** — 157/157 (156 + 1 new), 0 failed |

Test coverage: fail-closed defaults; `reset()` restoring them; finiteness for both
output modes; NaN-motor rejection; thrust mandatory even in `DirectMotors`;
level-bit composition (including that `None` is never supportable);
`supportsLevel()` masking for a partial-stack controller; the `reset_integrals`
contract being reachable; and `sizeof` budgets for the structs copied every gyro
cycle.

### Issue found and fixed

First run failed to link (`SMOKE_EXIT=2`) with undefined references to
`orb_publish`, `uorb_start`, `init_app_map` and friends, because I used
`px4_add_functional_gtest`. Switching to `px4_add_unit_gtest` fixed it, which is
also the in-tree convention for module-local tests (`PositionControl`,
`AttitudeControl`) and correct here: `ModuleParams` is header-only and nothing in
this test touches uORB or the parameter system at runtime.

**Correction (recorded during Stage 4).** I initially wrote that the functional
variant "does not link from a module subdirectory". That explanation is **wrong**
and was disproven at Stage 4: `VehicleState/` uses `px4_add_functional_gtest` from
a sibling module subdirectory and links fine. Re-testing `ControllerApi` under the
functional harness on a fully configured tree still fails, so the difference is
between the two targets, not their location — most plausibly that
`McControllerVehicleState` genuinely pulls in the parameter system while
`MulticopterControllerApi` does not, leaving the platform stubs unresolved.

The actionable rule, which holds in both observed cases: **use
`px4_add_unit_gtest` when the code under test needs nothing from the runtime, and
`px4_add_functional_gtest` when it uses `DEFINE_PARAMETERS`** (as
`mc_manual_mapping` at Stage 1 and `VehicleState` at Stage 4 both do).

**UNVERIFIED after Stage 3:** these are type definitions with no behaviour. No
state is produced, no command resolved, no control law run. The interface is
proven implementable and self-consistent — nothing more.

---

## Stage 4 — `VehicleState/VehicleStateProvider`

**Date:** 2026-07-29

Port of `MulticopterPositionControl::set_vehicle_states()` (velocity notch →
low-pass → derivative chain, position/velocity NaN semantics) plus the EKF-reset
bookkeeping from `::adjustSetpointForEKFResets()`, producing
`mc_ctrl::ControllerState`.

### Design decisions

- **Owns no uORB subscriptions.** The module feeds it messages, which is what
  makes it unit-testable without the uORB runtime and lets the tests drive
  synthetic EKF resets directly.
- **The first sample is never treated as a reset.** A fresh subscription arriving
  with non-zero reset counters would otherwise inject a spurious jump at every
  startup. Dedicated test.
- **`endCycle()` is explicit, not folded into `getState()`.** `getState()` returns
  a reference so it cannot clear afterwards, and clearing at cycle start would
  wipe resets latched earlier in the same cycle by `updateLocalPosition()`. The
  reasoning is in the header because it is the kind of thing a later reader would
  "simplify" and break.
- dt clamps replicate each stock stage exactly: position [0.002, 0.04],
  attitude [0.0002, 0.02], cycle [0.000125, 0.02].

### Gates

| Gate | Command | Result |
|---|---|---|
| BUILD (SITL) | `make px4_sitl_default` | **PASS** — exit 0 |
| BUILD (NuttX) | `make holybro_kakuteh7_default` | **PASS** — exit 0 |
| REGRESSION | `parameters.xml` vs baseline | **PASS** — diff empty |
| SMOKE | `make tests TESTFILTER=VehicleState` | **PASS** — `functional-VehicleStateProvider` |
| SMOKE | `make tests TESTFILTER=ControllerApi` | **PASS** — `unit-ControllerApi` |
| REGRESSION | `make tests` | **PASS** — 158/158 (157 + 1 new), 0 failed |

Test coverage: position pass-through and NaN-on-invalid; velocity chain settling
to DC with zero steady-state acceleration; filter reset on validity loss with a
bounded derivative on regain; first-sample-is-not-a-reset; reset deltas latched
once and cleared by `endCycle()`; velocity-reset filter carry-over producing no
derivative spike; quaternion reset latching; freshness flags with position at 1/10
gyro rate; dt clamping on large gaps; hover-thrust parameter fallback and
constraint.

### Harness bug found (not a code defect)

The Stage 4 gate script reported `SMOKE_EXIT=2` while the full suite passed
158/158 — contradictory, so it was investigated rather than accepted. Cause:
`make tests TESTFILTER="A|B"`. The Makefile expands `-DTESTFILTER=$(TESTFILTER)`
**unquoted**, so the shell interpreted `|` as a pipe
(`/bin/sh: 1: VehicleState: not found`) and additionally clobbered the test build
directory. **Regex alternation cannot be passed through `TESTFILTER`; run each
filter as a separate invocation.**

**UNVERIFIED after Stage 4:** filter *behaviour* is proven (DC pass-through, no
derivative spike across validity loss or velocity reset, dt clamps, one-shot reset
semantics). Numerical **equality with the stock mc_pos_control filter chain is NOT
proven here** — that arrives transitively at Stage 6's differential test against
the hand-wired `PositionControl` + `AttitudeControl` + `RateControl` triple, and
in flight at Stage 8.

---

## Stage 5a — `CommandFrontEnd/ControlLevelResolver`

**Date:** 2026-07-29

Stage 5 was split: 5a is level resolution (correctness-critical, small), 5b is the
~600-line Trajectory setpoint-sourcing port. Splitting keeps each gate meaningful
rather than bundling a proven-critical function with a large mechanical port.

`resolveLevel(vcm, vehicle_type, in_transition, is_tailsitter)` reproduces the
gating of all three stock modules. The design's §3 equivalence argument rests
entirely on this function.

### Why the test is trustworthy

`ControlLevelResolverTest` **links commander's own `mode_util` library** and calls
`mode_util::getVehicleControlMode()` for every `nav_state`, then applies
Commander's own `flag_multicopter_position_control_enabled` derivation
(`Commander.cpp:2652-2658`). It therefore cannot drift from commander, and it
fails loudly if upstream adds or changes a nav_state.

It also asserts the substitution the framework depends on, per nav_state rather
than in prose:

- mc_att_control's `manual && !altitude && !velocity && !position`
  ≡ `level == Attitude && manual`
- mc_rate_control's `manual && !attitude` ≡ `level == BodyRate && manual`

### Mutation testing (test validity)

A test that passes first try proves nothing unless it can fail. Two deliberate
mutations were injected and both were caught:

| mutation | result |
|---|---|
| drop the `flag_control_termination_enabled` guard | `TerminationAlwaysYieldsNone` **FAILED** |
| resolve Attitude before Trajectory | **3 FAILED**: truth table, offboard submodes, stick-gate equivalence |

Resolver restored and re-verified green afterwards.

### Gates

| Gate | Command | Result |
|---|---|---|
| BUILD (SITL) | `make px4_sitl_default` | **PASS** — exit 0 |
| BUILD (NuttX) | `make holybro_kakuteh7_default` | **PASS** — exit 0 |
| REGRESSION | `parameters.xml` vs baseline | **PASS** — diff empty |
| SMOKE | `make tests TESTFILTER=ControlLevelResolver` | **PASS** — `unit-ControlLevelResolver`, 8 cases |
| REGRESSION | `make tests` | **PASS** — 159/159 (158 + 1 new), 0 failed |

Coverage: 18-row nav_state truth table; all six OFFBOARD submodes (notably
`thrust_and_torque` → `None`, since it enables allocation without rates so the
framework must not claim control); termination and rates-disabled both → `None`;
fixed wing never reaching Trajectory; VTOL transition gating (tailsitter
mid-transition keeps the attitude loop, non-tailsitter does not); and that every
resolvable level except `None` is expressible in a `supportedLevels()` mask.

**UNVERIFIED after Stage 5a:** only the *level decision* is proven. No setpoint is
sourced, no takeoff ramp, no failsafe, no EKF-reset application to setpoints —
all of that is Stage 5b.

---

## Stage 5b — `CommandFrontEnd` setpoint sourcing

**Date:** 2026-07-29

The largest port in the plan: the Trajectory front-end out of
`MulticopterPositionControl::Run()` (failsafe, takeoff ramp, EKF-reset
application, constraints/`want_takeoff`, limit envelope, on-ground override),
plus Attitude and BodyRate sourcing wired to the Stage 1 stick mappings.

### Weaker evidence than other stages — stated up front

**There is no differential oracle here.** The stock logic is embedded in a module
`Run()` loop, not an extractable function, so unlike Stage 1 (and Stage 6) it
cannot be compiled as a verbatim twin. These tests assert documented stock
*semantics*, not bit-equality. **True equivalence arrives only at the Stage 8 SITL
A/B.** This is written into the test file header too, so a later reader does not
overestimate the coverage.

### Gates

| Gate | Command | Result |
|---|---|---|
| BUILD (SITL) | `make px4_sitl_default` | **PASS** — exit 0 |
| BUILD (NuttX) | `make holybro_kakuteh7_default` | **PASS** — exit 0 |
| REGRESSION | `parameters.xml` vs baseline | **PASS** — diff empty |
| SMOKE | `make tests TESTFILTER=CommandFrontEnd` | **PASS** — 14 cases |
| REGRESSION | `make tests` | **PASS** — 160/160 (159 + 1 new), 0 failed |

Coverage: NaN-per-axis pass-through; the 200 ms last-valid fallback boundary and
failsafe beyond it; both degraded-failsafe branches (blind-land when horizontal
velocity is unavailable; blind-descent at 0.3 m/s² when vertical is); EKF reset
applied exactly once; on-ground override with `reset_integrals`; takeoff-ramp
monotonicity; tilt limit slewing `MPC_TILTMAX_LND` → `MPC_TILTMAX_AIR`; Stabilized
generating **and publishing** an attitude setpoint; offboard attitude **consumed
but not republished** (the self-feedback path the design forbids); acro rate
generation; offboard NaN rate axes falling back to the measured rate; the
`reset_integrals` contract across level change / disarm / landing; termination
producing no command.

### Two test failures during development — both informative

1. **Five Trajectory tests returned NaN setpoints.** The test premise was wrong,
   not the code: the **on-ground override was correctly active**. Until
   `TakeoffState::flight`, the front-end substitutes an empty setpoint plus
   100 m/s² downward acceleration so no thrust is produced and no position
   corrections can wind up integrators pre-takeoff. Added a `takeOff()` helper
   that drives the state machine through spool-up and ramp exactly as the real
   vehicle does.
2. **Both failsafe tests.** After takeoff the setpoint is fresh, so failsafe
   legitimately never triggers. Reaching it requires leaving position control
   (re-arming the "no setpoint since entry" latch), re-entering with a stale
   setpoint, **and** being past the 200 ms window. Now explicit in the tests,
   which usefully documents how the two-stage failsafe actually engages.

### Build note

`TakeoffHandling::getTakeoffState()` is not `const`, so `CommandFrontEnd::takeoffState()`
cannot be either.

**UNVERIFIED after Stage 5b:** no control law has run. The command is assembled but
nothing consumes it, and equivalence with stock is asserted only semantically.

---

## Stage 6 — `CascadedPidController` (the reference implementation)

**Date:** 2026-07-29

Wraps the very same `PositionControl`, `AttitudeControl` and `RateControl` objects
the stock modules instantiate, reusing their `MPC_`/`MC_` parameters, expressed on
the pluggable interface. This is the A/B baseline: `MC_CTRL_ALG=1` should be
behaviourally indistinguishable from `MC_CTRL_ALG=0`.

### The differential test — strongest evidence in the project so far

`CascadedPidControllerTest` instantiates a **hand-wired stock triple** alongside
the controller and drives both from an identical scripted sequence, requiring
agreement to **1e-6f** on torque and thrust at every step:

- 3000 steps, mode sweep Trajectory → Attitude → BodyRate → Trajectory
- position samples at 1/10 gyro rate (exercises stage-rate parity)
- randomized attitude/velocity/rate wander so all three loops see real errors
- integral resets at each mode boundary
- an allocator saturation event mid-sequence

**Result: PASS.** This makes `MC_CTRL_ALG=1` numerically trustworthy before any
flight, and it retroactively strengthens Stage 4 — the state provider's filter
chain feeds the position stage, so a numerical divergence there would surface
here.

### Stage-rate parity

The plan called this "the single biggest fidelity risk". Handled: each stage gates
on `state.freshness` (`position_new` / `attitude_new`) and is fed its own `dt`
(`dt_position` / `dt_attitude` / `dt`). The differential test drives position at
1/10 gyro rate specifically to exercise it.

### Preserve-checklist items confirmed present

Yaw output LPF (`MC_YAW_TQ_CUTOFF`, with a dedicated test driving a Nyquist-rate
square wave), autotune additive rate injection, integral reset on disarm,
allocator anti-windup via `setAllocatorFeedback()`, `resetIntegralXY()` when
horizontal is uncontrolled, position/velocity gain and limit configuration
identical to `MulticopterPositionControl.cpp:198-204`.

### Gates

| Gate | Command | Result |
|---|---|---|
| BUILD (SITL) | `make px4_sitl_default` | **PASS** — exit 0 |
| BUILD (NuttX) | `make holybro_kakuteh7_default` | **PASS** — exit 0 |
| REGRESSION | `parameters.xml` vs baseline | **PASS** — diff empty |
| SMOKE | `make tests TESTFILTER=CascadedPid` | **PASS** — 5 cases incl. the differential |
| REGRESSION | `make tests` | **PASS** — 161/161 (160 + 1 new), 0 failed |

### Three defects found and fixed

1. `'MC_' does not name a type` — `MPC_*/MC_*` written inside a block comment; the
   `*/` terminated the comment early.
2. `-Werror=implicit-fallthrough` — the intentional cascade fallthrough needed
   explicit `[[fallthrough]]` markers.
3. Integrator test failed at 0.00078 ≠ 0. **The assertion was wrong, not the
   code**: `reset_integrals` zeroes the integrator and then the rate stage runs in
   the same cycle, legitimately re-accumulating one step — stock behaves
   identically. Asserting exact zero would have encoded a false expectation. Now
   asserts the wound-up history is gone (< 2% of prior magnitude).

**UNVERIFIED after Stage 6:** the controller has never run inside the module. No
uORB wiring, no output stage, no NaN guard or fallback latch exercised, nothing
published. All of that is Stage 7.

---

## Stage 7 — `MulticopterController` module glue + output stage

**Date:** 2026-07-29

`MulticopterController` runs as a `WorkItem` on the `rate_ctrl` work queue driven
by `vehicle_angular_velocity`, wiring
`VehicleStateProvider` → `CommandFrontEnd` → controller → output stage.
Adds `ControllerRegistry` (plain switch on `MC_CTRL_ALG`), `TemplateController`,
`module.yaml` (`MC_CTRL_ALG`, `MC_CTRL_EXT_ALC`, `MC_CTRL_WD_MS`) and
`template_controller_params.c` (`MC_TPL_*`).

### Live pipeline confirmed (disarmed SITL, stock modules stopped)

Stopping `mc_pos_control` / `mc_att_control` / `mc_rate_control` first is what makes
this a test rather than two publishers producing meaningless data.

```
vehicle_torque_setpoint   250 Hz   (== vehicle_angular_velocity 250 Hz)
vehicle_thrust_setpoint   250 Hz
mc_controller_status      update_count climbing, invalid_output_count 0,
                          fallback_active False, control_level 3 (Trajectory)
                          dt 0.004, dt_attitude 0.004, dt_position 0.008
```

`dt_position` correctly decoupled from `dt` — stage-rate parity holding in the
real module, not only in tests.

### Safety-net verification (the point of this stage)

`TemplateController` was temporarily patched to emit `torque(0) = NAN`, selected
via `MC_CTRL_ALG=2`, and run in SITL:

| observation | result |
|---|---|
| `fallback_active` | **True** |
| `fallback_reason` | **2 = INVALID_OUTPUT** |
| `mavlink_log_critical` | fired — "MC controller fallback: invalid controller output" |
| published `vehicle_torque_setpoint` | **finite**, no NaN |
| `mc_controller status` | `fallback : LATCHED (reason 2)` |
| patch reverted | cleanly |

### Bug found by chasing an inconsistent counter

The first run passed every stated assertion, but reported
`invalid_output_count: 1001` **equal to** `update_count: 1001`. By the control flow
those must diverge once the latch sets, so the anomaly was investigated rather
than logged green.

Cause: the disarm-clear was **level-triggered**:

```cpp
if (!_vehicle_control_mode.flag_armed && _fallback_latched) { _fallback_latched = false; ... }
```

Disarmed on the bench that clears the latch *every cycle*, re-runs the bad
controller, re-latches, and calls `latchFallback()` again — **~250 log lines per
second**, and the documented "sticky for the armed period" property silently did
not hold.

Fixed to be **edge-triggered** on the armed→disarmed transition (`_was_armed`).
Re-verified:

```
before fix:  update_count 1001, invalid_output_count 1001
after  fix:  update_count 1001, invalid_output_count 1      <-- one NaN caught, then latched
```

The protection itself was never broken — NaN was always caught and the reference
always produced valid finite output. The defect was in the latch's persistence and
logging.

### Other defects fixed

- `actuator_controls_status_s` has no `timestamp_sample` member.
- **CMake ordering trap:** `px4_add_module()` calls `get_target_property()` at
  *configure* time, and `src/modules/mc_controller` is processed before
  `mc_pos_control` defines `PositionControl` / `Takeoff`. Those cannot appear in
  the module's `DEPENDS`; they arrive transitively via the sub-libraries' `PUBLIC`
  linkage, which resolves at generate time. Commented in the CMakeLists so it is
  not "fixed" back.

### Design decisions

- Fallback latch sticky for the armed period; cleared only on the disarm edge.
- `MC_CTRL_ALG` changes **never allocate while armed** — deferred to next disarm
  with a warning, since `new`/`delete` on the RT path risks heap fragmentation.
- `_reference` is always allocated so fallback is instantaneous and
  allocation-free; when `MC_CTRL_ALG=1` the controller *aliases* it, and the
  destructor guards against a double free.
- On the first invalid output the reference is re-run **in the same cycle**, so the
  vehicle is never left without a command.

### Gates

| Gate | Command | Result |
|---|---|---|
| BUILD (SITL) | `make px4_sitl_default` | **PASS** — exit 0 |
| BUILD (NuttX) | `make holybro_kakuteh7_default` | **PASS** — exit 0 (after fix below) |
| SMOKE | disarmed SITL pipeline | **PASS** — 250 Hz publication, status streaming |
| SMOKE | forced-NaN fallback | **PASS** — latched, reason 2, finite output only |
| REGRESSION | `make tests` | **PASS** — 161/161, 0 failed |
| STYLE | astyle over all `mc_controller` sources | **PASS** (after fixing 2 test files) |
| PARAMS | new params present, `MC_CTRL_ALG` default 0 | **PASS** — `MC_CTRL_{ALG,EXT_ALC,WD_MS}`, `MC_TPL_{RATE_P,RATE_I,ATT_P}` |

`MC_CTRL_ALG` defaults to **0**, so this work changes nothing until deliberately
enabled.

### NuttX-only build break (why both targets are gated)

```
error: format '%u' expects argument of type 'unsigned int',
       but argument 4 has type 'uint32_t' {aka 'long unsigned int'}
```

`uint32_t` is `unsigned int` on x86_64 but `long unsigned int` on 32-bit ARM, so
`PX4_INFO("%u", _update_count)` compiled cleanly for SITL and failed
`-Werror=format=` on the Kakute H7. Fixed with an explicit `(unsigned long)` cast.
**A SITL-only workflow would have shipped a broken flight-board build.**

### Baseline note

`baseline_parameters.xml` had been wiped by the earlier `TESTFILTER` pipe
accident, so the param-delta gate could not run this stage. Verified instead that
the six new parameters exist with the correct `MC_CTRL_ALG` default of 0, and
re-baselined (19365 lines) for subsequent stages.

**UNVERIFIED after Stage 7:**
- **Fallback stickiness across an armed period is NOT proven** — the test runs
  disarmed, so that path was never exercised (and with the pre-fix code could not
  have been). Carried into Stage 8.
- Never armed, never flown with `MC_CTRL_ALG=1`. No A/B against stock.
- `rc.mc_apps` still starts the stock chain; the module was started by hand.
- The param-delta regression gate did not run this stage (baseline missing).

---

## Stage 8 — startup wiring + first flight + SITL A/B

**Date:** 2026-07-29

`rc.mc_apps` now branches on `MC_CTRL_ALG`. `param greater -s` is deliberate: `-s`
makes a missing parameter fail silently, so a build without
`MODULES_MC_CONTROLLER` always takes the stock branch.

### Boot directions

| `MC_CTRL_ALG` | mc_rate_control | mc_att_control | mc_pos_control | mc_controller |
|---|---|---|---|---|
| 1 | not running | not running | not running | **RUNNING** |

(The `=0` direction was checked in a run where the parameter had been left at 2 by
the Stage 7 NaN test, so that specific assertion did not execute; the stock branch
is however exercised by all four stock A/B flights, which boot with `=0`.)

### DEFECT 1 — intermediate topics were not published

`vehicle_rates_setpoint` and `vehicle_attitude_setpoint` were **entirely absent**
from the framework log (stock: 3745 and 1499 samples). At Trajectory level
`publishIntermediateTopics()` only published what the *front end* generated from
sticks, so nothing was published at all when the controller generated them.

Consumers per the design's §3 table: WeatherVane, mavlink `ATTITUDE_TARGET`,
gimbal stream, QGC. Fixed in `publishOutput()`, published only where the framework
*generated* the value, never where it consumed one (no self-feedback):
Trajectory → attitude + rate setpoint; Attitude → rate setpoint only; BodyRate →
neither.

### DEFECT 2 — trajectory yaw feed-forward was being dropped

At Trajectory level the attitude stage was called with `yaw_sp_move_rate = 0.f`.
Stock forwards `vehicle_attitude_setpoint.yaw_sp_move_rate`, which
`PositionControl.cpp:272` sets from the trajectory's commanded yaw rate. **The
framework silently lost trajectory yaw feed-forward**, so yaw lagged on every
commanded heading change.

**Stage 6's differential test passed anyway, because the hand-wired `StockTriple`
in that test contained the same mistake.** A reference twin written by the same
author reproduces the author's misunderstanding; 1e-6 agreement only proved two
copies of one wrong idea agreed. This is a structural limit of differential
testing against a self-written twin, and precisely why the SITL A/B against the
real stock modules was not redundant.

Fixed in both the controller and the test twin. Effect:

| signal | before fix | after fix |
|---|---|---|
| torque roll | 10.9% | 2.8% |
| torque yaw | 9.1% | 1.6% |
| rate sp roll | 7.7% | 1.8% |
| rate sp pitch | 5.0% | 1.2% |
| rate sp yaw | 8.2% | 1.9% |

### Statistical A/B — 4 stock flights vs 3 framework flights

A 2-run noise estimate proved unusable for low-amplitude torque signals: pitch
torque noise measured 1.9%, 3.0%, 4.3% and 7.4% across runs, so the derived
tolerance swung between 3.8% and 14.8% and the same result read as pass or fail
depending on which pair happened to be flown. With 6 pairwise stock samples,
**stock-vs-stock pitch torque varies by up to 17.1%.**

Final comparison (66 s aligned window):

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
itself as much as from stock. An earlier "roll DIFF" verdict was an artifact of
judging from a single framework flight.

Position tracking error vs commanded setpoint: within 1.6–2.7% on all axes.

### Framework health across a full armed flight

```
samples=6160  fallback_active_ever=no  invalid_output_count_max=0
```

**This closes the Stage 7 gap:** fallback stickiness across an armed period is now
exercised, with zero fallbacks and zero invalid outputs.

### Attitude level in flight — closed

The STABILIZED switch had been ACKed but never took (nav_state 15 absent from
every log), because PX4 requires a **live manual-control stream before** it will
accept a manual mode. **A COMMAND_ACK is not proof a mode engaged** — an earlier
claim that this gap was closed was based on the script printing "stick sweep
complete", which was not evidence.

Fixed in the profile: stream `MANUAL_CONTROL` first, then switch, then verify via
HEARTBEAT (`(custom_mode >> 16) & 0xFF == 7`) and skip the sweep loudly if it did
not engage. Verification flight (`s8_stab.ulg`):

```
nav_states:    [2, 4, 5, 14, 15, 18]        <- 15 = STAB present
control_level: {Attitude: 656, Trajectory: 5681}
fallback ever: False   invalid_max: 0
```

`StickToAttitudeSetpoint::update()` executed in situ and the Trajectory→Attitude
level transition was handled with zero fallbacks. **The Stage 1 coverage gap is
closed.**

**UNVERIFIED after Stage 8:**
- BodyRate/Acro level never flown (deliberate: scripted acro has high crash risk
  and negligible value over the unit tests, which cover the mapping bit-exactly).
- Nothing has run on real hardware.
- The `MC_CTRL_ALG=0` boot assertion did not execute directly (parameter was left
  at 2 by the Stage 7 test), though the stock branch is exercised by all four
  stock A/B flights.

---

## Stage 9 — TemplateController + README

**Date:** 2026-07-29

`README.md` documents the architecture, the 5-step process to add a controller,
the per-level input table, and the four rules that bite (honour
`reset_integrals`; always fill `output.thrust`, not just torque; keep
`update()` real-time safe; declare every level). It also records the limitation
of the Stage 6 differential test, since anyone extending this needs to know the
SITL A/B is not redundant.

| Gate | Result |
|---|---|
| `MC_CTRL_ALG=2` flies the full profile | **PASS** — 31874 updates, 0 invalid, no fallback |
| unsupported-level fallback | **PASS** — see below |

Unsupported-level path, forced by temporarily removing `Trajectory` from
`supportedLevels()`:

```
WARN [mc_controller] template does not support all control levels (0x06);
                     the reference cascade will take over for the rest
control_level: 3   fallback_active: True   fallback_reason: 1  (UNSUPPORTED_LEVEL)
status: fallback : LATCHED (reason 1)
```

Reverted cleanly; rebuild green. This is the second of the two fallback triggers
(Stage 7 covered `INVALID_OUTPUT`), and the one that matters directly for a
rate-only FTC allocator.

---

## Stage 10 — `CA_EXT_ALLOC` + direct-motors path — **SUPERSEDED**

**Date:** 2026-07-29

> Reverted in full. The direct-motors output path, `CA_EXT_ALLOC`,
> `MC_CTRL_EXT_ALC` and `OutputType` are gone, and `control_allocator` is stock
> again. A static parameter cannot make a second `actuator_motors` publisher safe:
> the fallback swaps the direct-motors controller for the torque-output reference
> cascade mid-flight, at which point nothing publishes the topic and the ESCs hold
> their last value. See ARCHITECTURE.md §3.8. The gates below stand as a record of
> what was measured — note that none of them exercised the fallback.

- `control_allocator` gains `CA_EXT_ALLOC`, gating `_publish_controls` on
  `!(CA_EXT_ALLOC && flag_armed)`. Gated on **armed** deliberately: while
  disarmed the allocator keeps publishing so `actuator_test` and ESC calibration
  still work on the bench — exactly when FTC work needs them.
- `DirectMotorController` (`MC_CTRL_ALG=3`) **subclasses** `CascadedPidController`
  and overrides only the allocation with a quadrotor-X pseudo-inverse. The outer
  loops stay verified; only the mixing changes. This is the shape a
  failure-tolerant allocator takes.

| Gate | Result |
|---|---|
| controller identity | **PASS** — `direct_motors`, `output type: actuator_motors` |
| flew the full profile | **PASS** — 33616 updates, 0 invalid, no fallback |
| **auto-land completed and DISARMED** | **PASS** — proves `vehicle_thrust_setpoint` still feeds land_detector through the bypass |
| `actuator_motors` values | **PASS** — 1297/1297 finite, range [0.000, 1.000] |
| `actuator_motors` at gyro rate | **PARTIAL** — the log shows 10 Hz, but that is the *logger's* per-topic downsample, not the publication rate. `uorb top` was sampled while disarmed, so the armed publication rate was not measured. The flight succeeding is indirect evidence only. |

---

## Stage 11 — fault-injection equivalence — **NOT VERIFIED**

**Date:** 2026-07-30

**Goal:** confirm the framework fails the same way as stock under a rotor loss, so
that `MC_CTRL_ALG=0` vs `=1` is a valid control pair when evaluating an FTC
controller.

**Outcome: no rotor was ever killed. The stage tested nothing.**

### First attempt — silently tested nothing

Both arms flew, both printed `PROFILE COMPLETE`, both logs were captured, exit 0.
Buried in the output: `ERROR [failure] Failure type '1' not found`. The syntax is
`failure <unit> <type> -i <instance>`; `failure motor 1 off` put the instance
where the type belongs.

### Second attempt — command accepted, injection still did not occur

Syntax corrected to `failure motor off -i 1`, which reported no error
(`INJECTION_framework=issued`). ULog analysis nonetheless shows no failure in
either arm:

```
stock:      motor_failure_mask max=0, nonzero_samples=0, all 4 motors reach 1.000
framework:  motor_failure_mask max=0, nonzero_samples=0, all 4 motors reach 1.000
```

`FailureInjector` (`commander/failure_detector/FailureInjector.cpp:50`) never
logged `CMD_INJECT_FAILURE`, so it never processed the command — most plausibly
because `SYS_FAILURE_EN` was set at runtime and
`_failure_injection_enabled` had not picked it up. The x500 model does load
`MotorFailurePlugin`, so the simulator side is present.

### The pattern this stage shares with Stages 7 and 8

Three times in this build a green result was not evidence:

| stage | reported | actually |
|---|---|---|
| 7 | every NaN assertion passed | latch re-triggering every cycle, logging at 250 Hz |
| 8 | script printed `stick sweep complete` | STABILIZED never engaged (nav_state 15 absent) |
| 11 | both profiles completed, logs captured | injection rejected, then silently not applied |

**A script's success message is not evidence the thing under test happened.**
Every SITL gate needs an independent check in the ULog or a status topic.

### What can and cannot be claimed

**Cannot:** that the framework and stock respond identically to a rotor loss.

**Available argument, not a measurement:** the branch's `kill_switch_2` injector
lives in `mixer_module`, **downstream of `actuator_motors`**, so it is
architecturally independent of which controller produced the moments — and Stage 8
established that `actuator_motors`' upstream inputs (torque/thrust) are
statistically indistinguishable between the two arms. That makes equivalence
likely, but it is reasoning, not data.

### How to actually close it

1. Set `SYS_FAILURE_EN=1`, `param save`, **reboot**, then inject — rather than
   setting the parameter at runtime.
2. Or, preferred for this project: verify with the branch's own `kill_switch_2`
   on hardware with the RC transmitter, alongside the tethered-hover step. That
   path is the one that actually flies on the airframe, and it cannot be driven
   over MAVLink because it reads `manual_control_switches` from real RC.

---

## Stage 12 — stock work-queue separation

**Date:** 2026-07-29 / 2026-07-30 UTC
**Goal:** restore stock's two-work-queue split so heavy trajectory math stays off
the gyro path, *without* breaking monolithic full-stack controllers — which cannot
be cut across two threads and are the framework's whole reason to exist.

### Design constraint that shaped this

The user's controllers run "from desired position / roll / pitch rate all the way
to motor moments". A law whose stages share internal state must stay in one
`update()` call. So the split is **opt-in**, via three additive virtual methods
that default to the existing behaviour:

```cpp
virtual bool hasOuterStage() const { return false; }
virtual bool updateOuter(const ControllerState &, const ControllerCommand &,
                         float, vehicle_attitude_setpoint_s &) { return false; }
virtual void fillLocalPositionSetpoint(vehicle_local_position_setpoint_s &) const {}
```

`hasOuterStage() == false` → `OuterLoop::Run()` returns immediately, the whole law
runs at gyro rate, byte-for-byte the pre-Stage-12 behaviour. Nothing an existing
controller does had to change.

### Gates

| gate | command | result |
|---|---|---|
| BUILD SITL | `make px4_sitl_default` | ok |
| BUILD NuttX | `make holybro_kakuteh7_default` | ok |
| TESTS | `make tests` | **161/161** |
| style | `make check_format` | clean |
| queue placement | `work_queue status` | see below |
| flight, split | `PROFILE_STAB=1 flight_profile.py` ×3 | complete, no fallback, 0 invalid |
| publication matrix | `pub_matrix.py <stock> <split>` | matches after the fix below |

**Queue placement — the thing this stage exists to change:**

```
STOCK   wq:rate_ctrl            mc_rate_control       250.0 Hz  4000 us
        wq:nav_and_controllers  mc_att_control        250.0 Hz  4000 us
                                mc_pos_control        125.0 Hz  8000 us
SPLIT   wq:rate_ctrl            mc_controller         250.0 Hz  4000 us
        wq:nav_and_controllers  mc_controller_outer   125.0 Hz  8000 us
```

Inner/outer update counts across a flight: 33341 / 14852 = ratio 2.24 ≈ 250/125.
Measured from `mc_controller_status.dt` over 3 flights: inner 250.0 Hz
(mean = median = p1), attitude stage 247.6–249.6 Hz, position stage 125.0 Hz.

### DEFECT 1 — outer item passed `now = 0`; the vehicle never left the ground

`ControllerState::timestamp_sample` is written by `updateAngularVelocity()`. The
outer item is driven by `vehicle_local_position` and never calls it, so the
timestamp stayed 0. `TakeoffHandling` reads 0 as "now", never left rampup, and the
climb-rate limit never opened — thrust pinned at −0.001, `max_alt` 0.0 m.

Fix: `VehicleStateProvider::setTimestampSample()`, called from `OuterLoop::Run()`
with `local_position.timestamp_sample`.

**Why no gate caught it earlier:** every unit test drives the state provider
through `updateAngularVelocity()`, because until this stage every consumer did.

### DEFECT 2 — `vehicle_local_position_setpoint` published in every mode

Found by the A/B, and it is the most instructive failure in this log.

The Stage 12 A/B (5 stock vs 3 split flights) reported torque, thrust and
rate-setpoint differences all inside run-to-run noise. A separate check of
**position tracking error** did not:

```
                    x        y        z      [m RMS, in-air 66 s window]
stock   n=5      1.051    1.067    0.120
split   n=3      1.086    1.112    1.497    <-- 12x worse, and reproducible +/-0.005
```

Localising it by thirds of the window put the entire error in the last third — the
Stabilized segment — while the first two thirds matched stock. Counting samples per
flight mode found the cause:

```
during STABILIZED:   stock  local_position_setpoint samples = 0    (correct)
                     split                                  = 146  at a frozen z = -4.97
```

[MulticopterController.cpp:540] published the topic unconditionally whenever the
reference controller was active — ignoring both the control level and which work
item generated it. Stock only publishes it while `mc_pos_control` runs, and
`mc_pos_control` does not run outside a position mode. Two distinct bugs:

1. **Wrong modes** — published through Stabilized and Acro, at gyro rate. This is
   *pre-existing*, not introduced by Stage 12; it was never exposed because the
   Stage 8 framework flight never entered Stabilized.
2. **Duplicate publisher** — in split mode `OuterLoop` publishes it too.

The vehicle flew correctly throughout (actual altitude 4.96 → −0.42 m in STAB
against stock's 4.99 → −0.09 m). The consequence is a stale reset origin handed to
the flight tasks, and log analysis that lies.

Fix: gate it exactly as the attitude setpoint three lines above already was —
`_last_level == Trajectory && !outerStageActive()`.

**Verification:**

```
                    z-RMS      thirds of window        max
stock               0.123    0.097/0.023/0.015        0.99
split, before       1.491    0.081/0.020/1.334        5.44   (x3 flights, +/-0.007)
split, after        0.113    0.083/0.025/0.011        1.00   (x2 flights)
publication matrix: none - matches stock in every mode  (x2 flights)
```

### The methodological lesson

**Comparing values is not comparing behaviour.** Stage 8 and Stage 12 both
compared signal *magnitudes* and both passed. Neither could see a topic published
in the wrong flight mode, because the numbers inside it were entirely plausible —
a frozen −4.97 m altitude setpoint looks exactly like a real one.

The gate that catches this class asks a different question: **which topics are
published in which modes.** `pub_matrix.py` counts samples of every intermediate
topic per control-level segment and diffs stock against the framework. It is now
part of the Stage 8/12 gate set. The plan called for this assertion in §3
("Publish/consume rule — assert in review"); it was applied to
`vehicle_attitude_setpoint` and `vehicle_rates_setpoint` and simply missed for
`vehicle_local_position_setpoint`. Reviewing the rule three lines above the defect
would have found it.

This is the fourth "false green" in this log, and the second where a green A/B hid
a real defect. Running it: `python3 pub_matrix.py <stock.ulg> <framework.ulg>` —
the flight must include a Stabilized segment (`PROFILE_STAB=1`), or the modes
where the defect lives are never entered.

### Statistical A/B, 5 stock vs 3 split flights

```
signal            within-stock     within-fw       between    verdict
torque roll        9.1 ( 4.2)     6.8 ( 4.5)    8.8 ( 4.1)   noise-dominated
torque pitch      17.1 ( 8.7)    11.4 ( 7.6)   22.9 ( 8.7)   noise-dominated
torque yaw        17.1 ( 7.6)     0.9 ( 0.6)   17.1 (12.3)   noise-dominated
thrust z           0.2 ( 0.1)     0.0 ( 0.0)    1.9 ( 1.8)   flagged
rate sp roll       9.2 ( 4.3)     2.7 ( 1.8)    9.0 ( 4.6)   noise-dominated
rate sp pitch      5.7 ( 2.6)     2.5 ( 1.7)    9.0 ( 5.8)   flagged
rate sp yaw       15.5 ( 6.9)     0.9 ( 0.6)   16.3 (11.7)   flagged
                                        [% max (mean) pairwise RMS difference]
```

Three signals sit outside run-to-run noise. `thrust z` is the credible one: both
groups are internally near-perfect (0.2% and 0.0%) with a consistent 1.9% offset
between them, ~10x either group's own spread. `rate sp pitch/yaw` are weaker — the
between-group figure stays inside stock's own max spread.

Note the split flights are *more* self-consistent than stock (0.0–2.7% vs
0.2–17.1%), as expected from a deterministic uORB handoff, which is also what makes
a small systematic offset visible at all.

**UNVERIFIED:** the cause of the 1.9% `thrust z` offset. It does not bias the
user's own experiments — those compare framework+reference against
framework+their-controller, with stock outside the loop — but it is a real
framework-vs-stock fidelity gap and is not explained.

**UNVERIFIED:** cycle times. Every `PC_ELAPSED` counter reports `0us elapsed` in
SITL, for the stock modules too. The 2.5 ms H7 inner-loop budget at
`IMU_GYRO_RATEMAX=400` can only be checked on hardware.

**UNVERIFIED (by construction):** the disjoint-state contract between
`updateOuter()` and `update()`. It is enforced by review, not the compiler. Only
`CascadedPidController` opts in today — `_position_control` touched solely by the
outer stage, `_attitude_control` / `_rate_control` solely by the inner.

### Procedural errors in this stage, recorded so they are not repeated

1. **Overlapping SITL runs** — a second script's `pkill` killed the first run's
   simulator; `loop_perf` came out 0 bytes. All SITL work must be strictly serial.
2. **Stray `px4` process** → 29 test failures (every test that spawns px4). Clean
   re-run: 161/161. Always `pgrep -f bin/px4` before trusting a test result.
3. **Missing `param save` + reboot** → both A/B arms flew stock. `MC_CTRL_ALG` is
   `reboot_required` and `rc.mc_apps` branches at boot; a runtime `param set`
   changes nothing. Always confirm the arm from `mc_controller status`.
4. **Monitor grep pattern excluded `TIMEOUT`** → reported "full flight, zero
   fallbacks" while the log said `[profile] TIMEOUT during climb`. Filtering *for*
   success is itself the bug. Patterns now include `TIMEOUT|FAILED|REJECT`.

### DEFECT 3 — the manual throttle curve never received the hover-thrust estimate

Found by chasing the one signal that survived the Stage 12 A/B. Six of seven
channels were noise-dominated; `thrust z` was not (1.9% between groups against
0.2% within stock and 0.0% within the framework). That 1.9% turned out to be a
21% error, diluted.

Stock feeds the estimate to **two** consumers:

| consumer | stock call site | purpose |
|---|---|---|
| `PositionControl` | `MulticopterPositionControl.cpp` | trajectory-level collective |
| `StickToAttitudeSetpoint` | `mc_att_control_main.cpp:119-124` | stick → collective throttle curve |

The framework wired only the first, via `VehicleStateProvider` →
`state.hover_thrust`. `CommandFrontEnd` contained **no** hover-thrust reference at
all, so the manual throttle curve fell back to the `MPC_THR_HOVER` parameter.

On the SITL x500 the parameter is 0.60 and the live estimate is 0.72 — a 20.9%
deviation, because the default parameter simply does not match the airframe. The
commanded collective in Stabilized:

```
stock                      -0.7257                        (live estimate 0.7253)
framework, before          -0.6000  -0.6000  -0.6000      <-- exactly MPC_THR_HOVER, x3 flights
framework, after           -0.7258  -0.7256  -0.7257      <-- matches stock to 2e-4, x3 flights
```

Three flights landing on precisely the parameter value is proof rather than
inference; three more landing on stock's value to four decimals closes it.

Fix: `CommandFrontEnd::setHoverThrustEstimate()` forwarding to
`StickToAttitudeSetpoint`, called from `MulticopterController::pollInputs()`
alongside the existing state-provider call. The outer item needs no equivalent —
it only ever handles Trajectory level, and the attitude setpoint its front end
computes at other levels is discarded.

**Why the aggregate metric hid it.** Stabilized is ~15 s of the 66 s comparison
window, and the Trajectory segments (~77% of samples) were always correct because
`state.hover_thrust` did reach `PositionControl`. A whole-window RMS averaged a 21%
single-mode error down to 1.9% — close enough to the noise floor that two of the
three flagged signals in the same table are genuinely marginal.

**This is the lesson of Stage 12, stated properly:** a whole-flight aggregate can
mask a large error confined to one flight mode. Per-mode analysis is not optional,
in either direction — `pub_matrix.py` segments by control level for the same
reason. Had the A/B reported only "6 of 7 signals within noise", the defect would
have shipped to hardware, where a pilot switching to Stabilized would have met a
sudden ~21% collective drop.

**Permanent regression test:**
`CommandFrontEndTest.ManualThrottleCurveFollowsHoverThrustEstimate` drives the
front end at mid-stick under two hover-thrust estimates and asserts the collective
moves. Mid-stick is the discriminating input — at the stick extremes the curve is
pinned to `MPC_THR_MIN`/`MPC_THR_MAX` and the estimate cannot be observed, which is
part of why no existing test caught this.

### The mutation test that rejected its own fix

The first fix for DEFECT 3 added `CommandFrontEnd::setHoverThrustEstimate()` and
called it from `MulticopterController::pollInputs()`, alongside a new unit test.
Test green, three flights matching stock to 2e-4. It looked finished.

Then the mutation test — delete the wire, confirm the test goes red:

```
wire removed from MulticopterController.cpp -> make tests TESTFILTER=CommandFrontEnd
  100% tests passed, 0 tests failed out of 1        <-- WRONG. Should have failed.
```

**The test could not possibly have caught the defect.** It called
`fe.setHoverThrustEstimate()` directly, so it verified the front-end pass-through —
while the actual defect was a *missing call site in module glue*, which no
`CommandFrontEnd` unit test can observe. The test guarded a different link in the
chain than the one that broke.

Worse, on investigation the test was passing for a spurious reason even in the
correct build. Two problems:

1. It reused one `CommandFrontEnd` across both cases via `reset()`. The
   `_manual_throttle_minimum`/`_maximum` slews evolve during a run and survive
   `reset()`, so spool-up state leaked from the first case into the second and
   produced a difference unrelated to hover thrust.
2. It ran 200 x 4 ms = 0.8 s. The estimate reaches the curve through a **0.05/s**
   slew (`StickToAttitudeSetpoint.cpp:50`), which cannot traverse a meaningful
   thrust delta in 0.8 s regardless of wiring.

So: a green test, a green three-flight A/B, and the test was measuring slew-state
leakage.

**The response was to change the design, not to write a better test.** The value is
now derived inside `CommandFrontEnd::update()` from `state.hover_thrust`, which the
front end already receives every cycle:

```cpp
_stick_to_attitude.setHoverThrustEstimate(state.hover_thrust_valid ? state.hover_thrust : NAN);
```

The public setter is gone, and so is the second call site the module had to
remember. `VehicleStateProvider::setHoverThrustEstimate()` is now the single entry
point, and the two consumers cannot diverge. This also makes the path reachable
from a unit test, because a test driving `update()` supplies the state.

`NAN` when invalid rather than the parameter value: the mapping keeps its own slew
and fallback, matching stock.

**Re-run of the mutation test, on the rewritten test (fresh instance per case,
2000 x 4 ms = 8 s):**

```
(a) with the fix          [       OK ]
(b) forwarding deleted    [  FAILED  ]   low(0.35)=-0.5  high(0.65)=-0.5
(c) restored              [       OK ]
```

Both cases collapse to the harness `MPC_THR_HOVER` when the path is cut, and the
assertion prints the values so a future failure explains itself.

Then: style clean, **161/161**, SITL and NuttX builds clean, and three more flights
on the refactored path.

### What this stage says about verification, in one place

Five separate green results in this project were not evidence: three scripts whose
success messages did not mean the thing under test happened (Stages 7, 8, 11), one
statistical A/B that passed while a topic was published in the wrong mode
(Stage 12), and one unit test that passed while measuring the wrong quantity
entirely (this section).

The generalisable rules:

1. **A test you have never seen fail is not yet evidence.** Mutate the code it
   guards and watch it go red. This is the only gate here that caught a bad gate.
2. **If a defect lives in glue that no unit test can reach, change the design so
   it does not.** Deriving a value from state the function already receives beats
   remembering to push it in — the second call site is the bug waiting to happen.
3. **A green aggregate metric is not equivalence.** Segment per flight mode and
   check publication per mode (`pub_matrix.py`).
4. **Verify the arm actually flew.** `mc_controller_status` present/absent in the
   ULog distinguishes framework from stock independently of what any script claimed.

### Final verification of the state-derived refactor

Three flights on the refactored path, against three stock flights with a Stabilized
segment (all six confirmed by `mc_controller_status` present/absent in the ULog, not
by what a script claimed):

```
STAB collective     stock  -0.7259  -0.7259  -0.7257
                refactored -0.7256  -0.7256  -0.7257      match within 3e-4

publication matrix  matches stock in every mode, x3 flights
per-mode A/B        REAL: nothing outside run-to-run noise
  Trajectory        every substantive signal noise-dominated
  Attitude          thrust z (RMS 0.7248) 0.0% between groups
```

Gates: style clean, **161/161** tests, SITL and NuttX builds clean.

### The A/B tooling needed three fixes of its own

Worth recording, because a bad gate is more dangerous than no gate - it converts
absence of evidence into a false all-clear.

1. **`max`-vs-`max` was statistically biased.** With n flights per arm the
   between-group set has n*n pairs and each within-group set has n*(n-1)/2 - 9
   versus 3 at n=3. A max over more pairs is systematically larger, so the rule
   flagged nearly everything, including a **0.1%** difference. Now compares means
   (unbiased with respect to pair count) with a 1.5x factor, keeping the max test
   only as a secondary condition.
2. **No relative floor.** Added `FLOOR_PCT = 2%`: below that a difference is not
   actionable whatever the ratio.
3. **No absolute floor.** This was the subtle one. During a Stabilized stick sweep
   the yaw rate setpoint sits at ~0.0005 rad/s (0.03 deg/s) between inputs, and the
   Attitude-level torques at ~0.0001 normalized. A 47% relative difference there is
   0.0002 rad/s and says nothing about the controller. Signals below `MAG_FLOOR`
   are now labelled `negligible (|signal| < x)` **with their magnitude printed**,
   rather than silently passing the ratio test or being flagged as real.

The `|signal|` RMS column exists because of (3): without it, a 60% spread on a
numerically-zero signal looks alarming and a reviewer cannot tell the difference.

### Chain tracing

`tools/trace_chain.py` autogenerates the input -> module -> output tree from a
ULog, with per-edge rates, trigger source and work queue.
`tools/trace_live.sh` snapshots the running system for what a log cannot provide.

The rate labelling is the load-bearing part. The logger downsamples most chain
topics (`logged_topics.cpp`), so counting ULog samples measures the *logger*:
`vehicle_angular_velocity` is capped at 20 ms and `actuator_motors` at 100 ms, which
is why a 250 Hz inner loop reads as 50 Hz and the motors as 10 Hz. Both numbers
misled this project before the caps were accounted for. Every rate is therefore
labelled `[measured]` (from the module's own `dt`), `[capped]` (with the cap shown),
`[on-change]` (verified uncapped) or `[logged]` (**cap unknown - the tool says so
rather than guessing**). True rates for capped topics come from `uorb top -1` on the
live system only.
