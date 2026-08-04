# Adding and testing a controller

Two things, in the order you will actually do them: get your law running, then
prove it works. Nothing here needs me — every command is reproducible.

---

## Part 1 — Add a controller

### The three files you write

```
controllers/MyController.hpp        # class declaration
controllers/MyController.cpp        # YOUR MATH
controllers/my_controller_params.c  # your gains as MC_MY_* parameters
```

Start by copying the template, which is a working (if crude) full-stack controller:

```bash
cd src/modules/mc_controller/controllers
cp TemplateController.hpp MyController.hpp
cp TemplateController.cpp MyController.cpp
sed -i 's/TemplateController/MyController/g; s/"template"/"mine"/' MyController.{hpp,cpp}
```

### The four one-line edits

| file | edit |
|---|---|
| `controllers/ControllerRegistry.hpp` | add `MyLaw = 4` to `enum class Algorithm` |
| `controllers/ControllerRegistry.cpp` | add `case Algorithm::MyLaw: return new MyController(parent);` |
| `controllers/CMakeLists.txt` | add `MyController.cpp` to `SRCS` (**not** `my_controller_params.c` — `src/lib/parameters` globs `src/*params.c` and generates it) |
| `module.yaml` | add `4: My law` under the `MC_CTRL_ALG` enum |

`-Wswitch` will fail the build if you add the enumerator and forget the case.

### Where your math goes

One function: `MyController::update()`. It is handed the state and the command
and must fill the output. `command.level` tells you what was commanded:

```cpp
bool MyController::update(const ControllerState &state,
                          const ControllerCommand &command,
                          float dt, ControllerOutput &output)
{
    if (command.reset_integrals) {        // HARD CONTRACT - always honour this
        _integral.setZero();
    }

    switch (command.level) {
    case ControlLevel::Trajectory:        // position/velocity/accel + yaw, in NED
        // command.position_sp, .velocity_sp, .acceleration_sp, .yaw_sp
        // command.thrust_min/max, .tilt_limit, .vel_limit_xy/up/down
        break;
    case ControlLevel::Attitude:          // command.attitude_sp (unit quaternion)
        break;
    case ControlLevel::BodyRate:          // command.rate_sp (guaranteed finite)
        break;
    case ControlLevel::None:
        return false;
    }

    output.torque = ...;                  // normalized [-1,1], body FRD
    output.thrust = ...;                  // REQUIRED, not just torque
    output.valid  = true;                 // without this the framework rejects you
    return true;
}
```

**If your law is monolithic** — position through rates in one pass, sharing state —
delete the `switch` entirely and branch only on `command.level` to decide which
setpoint to read. Leave `hasOuterStage()` false and the whole law runs at gyro
rate on one thread. This is the default and the case to build first.

Inputs available on `state`: `q`, `angular_velocity`, `angular_accel`, `position`,
`velocity`, `acceleration`, `heading`, `hover_thrust`, the `*_valid_*` flags,
`landed` / `maybe_landed` / `armed` / `spooled_up`, `resets` (EKF discontinuities),
and `freshness` (per-stage `dt`, plus `position_new` / `attitude_new` so you can
gate slow stages).

### Build and select

```bash
make px4_sitl_default
# in the pxh console, or over MAVLink:
param set MC_CTRL_ALG 4
param save
reboot                      # MC_CTRL_ALG is reboot_required
mc_controller status        # must print: active controller: mine
```

`param set` alone does nothing — `rc.mc_apps` branches on this parameter at boot.
**Always confirm from `mc_controller status`**, never assume; flying the wrong arm
and not noticing has happened in this project.

---

## Part 2 — Test it

Four rungs, cheapest first. Do not skip upward — each one catches things the next
cannot see.

### Rung 1 — Unit test (seconds, no simulator)

Write a test next to your controller and register it with
`px4_add_unit_gtest(SRC MyControllerTest.cpp LINKLIBS ...)`. Drive `update()` with
synthetic state and assert on the output. Minimum worth asserting:

```bash
make tests TESTFILTER=MyController
make tests                       # full suite, expect 161/161 (+ yours)
```

- output is finite and `valid` for every level you declare in `supportedLevels()`
- `command.reset_integrals` actually zeroes your integrators
- a NaN or absurd input does not produce NaN output
- **your test fails when you break the code.** A test you have only seen pass is
  not yet evidence. Change a sign, confirm red, change it back.

### Rung 2 — Does it fly at all (one SITL flight)

```bash
HEADLESS=1 make px4_sitl gz_x500
# separate terminal, once "Gazebo world is ready":
PROFILE_STAB=1 python3 src/modules/mc_controller/tools/flight_profile.py
```

Flies arm → takeoff → position box → Stabilized stick sweep → land. Then check
**from the log and status, not from the script's output**:

```bash
build/px4_sitl_default/bin/px4-mc_controller status
#   active controller : mine          <- the right arm actually flew
#   fallback          : no (reason 0) <- your output was never rejected
#   invalid outputs   : 0
```

`PROFILE_STAB=1` matters: it is what enters Stabilized, and two of the three
defects found in this framework lived in modes a position-only flight never
reaches.

### Rung 3 — Compare against the reference (the real gate)

`MC_CTRL_ALG=1` is the stock cascade on this same interface, so it is a like-for-like
baseline. Fly **at least 3 flights per arm** — SITL is not deterministic, and n=1
comparisons produced a phantom regression in this project twice.

```bash
# 3 flights with MC_CTRL_ALG=1, 3 with MC_CTRL_ALG=4, saving each ULG. Then:
cd src/modules/mc_controller/tools

# a) signal magnitudes, within-group vs between-group spread
python3 group_compare.py --stock ref1.ulg ref2.ulg ref3.ulg --fw mine1.ulg mine2.ulg mine3.ulg

# b) which topics get published in which flight modes
python3 pub_matrix.py ref1.ulg mine1.ulg
```

**Run both.** They are not redundant — each catches a class the other
structurally cannot:

| tool | catches | missed by |
|---|---|---|
| `group_compare` | your law commanding different moments | — |
| `pub_matrix` | a topic published in the wrong mode (numbers look plausible, so magnitudes pass) | `group_compare` |

**Currently uncovered: per-mode segmentation.** A large error confined to one
flight mode is diluted by a whole-flight average — a 21% collective error in
Stabilized was once reported as 1.9% that way. The tool that segmented by control
level was removed: its fixed 66 s window clipped the Stabilized sweep in half,
manufacturing an 18.6% phantom pitch-torque difference that measures 1.2% over
the true segment, and silently dropped the Attitude level entirely when the sweep
fell outside the window. Until it is rewritten — window derived from the segment
boundaries, not a fixed offset — segment by hand before trusting a green aggregate.

How to read them: a difference is only real when the **between-group** spread
exceeds **both** within-group spreads. Stock-vs-stock pitch torque varies up to
17% run to run, so a flat "within 5%" tolerance is unusable. `group_compare`
needs at least 2 flights per arm to have a within-group spread at all; on n=1 it
fails with a `max() iterable argument is empty` traceback rather than saying so.

### Seeing the whole chain — what goes where, and how often

Two views, because neither alone is honest.

**Post-flight, from a ULog.** Autogenerates the full tree from gyro to ESCs — each
module's inputs, outputs, trigger, work queue, and rate:

```bash
python3 tools/trace_chain.py flight.ulg            # the tree
python3 tools/trace_chain.py flight.ulg --values   # + signal magnitudes (RMS)
python3 tools/trace_chain.py flight.ulg --mode Attitude   # one control level only
```

```
├─ MODULE mc_controller  [inner]
│   queue    : wq:rate_ctrl
│   trigger  : vehicle_angular_velocity
│     IN  vehicle_attitude_setpoint      125.0 Hz  [measured]  from mc_controller_status dt
│     OUT vehicle_torque_setpoint        250.0 Hz  [measured]
│                                        RMS xyz[0]=0.0139 xyz[1]=0.0108 xyz[2]=0.0497
│     OUT actuator_motors                 10.0 Hz  [capped]  logger cap 100 ms = 10 Hz max
```

**Read the rate labels — this is the part that matters.** The logger downsamples
most of these topics, so counting samples in a ULog measures *the logger*, not your
software:

| label | meaning |
|---|---|
| `[measured]` | the module's own loop `dt` from `mc_controller_status` — a real rate |
| `[capped]` | logger-limited; true rate is **higher**, and the cap is printed |
| `[on-change]` | verified uncapped, so a low rate is genuinely event-driven |
| `[logged]` | cap unknown — the tool says so rather than guessing |

`vehicle_angular_velocity` is capped at 20 ms and `actuator_motors` at 100 ms, so a
naive reading shows a 250 Hz inner loop as 50 Hz and the motors as 10 Hz. Both
numbers misled me during this project before the caps were accounted for.

**Live, from the running system** — the only source of true publication rates:

```bash
./tools/trace_live.sh                 # one snapshot, while armed and flying
./tools/trace_live.sh -n 5 -i 10      # 5 snapshots 10 s apart
```

It reports `mc_controller status` (which controller is *actually* active),
`work_queue status` (what runs where), `uorb top -1` (true rates), `perf` (cycle
time — hardware only), and **publisher counts per output topic**. That last one
catches a whole class of bug directly: two publishers on one setpoint topic is
always wrong, and it has happened in this framework twice.

### Rung 4 — Hardware

Some things are only measurable here. **Cycle time in particular: every
`PC_ELAPSED` counter reads `0us elapsed` in SITL**, for the stock modules too, so
SITL can tell you nothing about whether your law fits the budget.

```bash
# bench, props OFF, disarmed:
mc_controller status
work_queue status        # confirm your item's queue and rate
perf                     # mc_controller: cycle  <- the number that matters
```

Budget: at `IMU_GYRO_RATEMAX=400` (the hardware default; SITL uses 250) the inner
loop has **2.5 ms**. If your law does not fit, either optimise it or split it —
see `hasOuterStage()` in [README.md](README.md), which moves the trajectory stage
to the slower queue.

Then, props on and tethered: Stabilized hover → Altitude → Position → mission.
`param set MC_CTRL_ALG 0` + reboot is one step away at every point.

---

## Testing fault tolerance specifically

Your `kill_switch_2` in `mixer_module` sits **downstream of `actuator_motors`**, so
it is independent of which controller produced the moments — the same injection
works for any `MC_CTRL_ALG`. That is what makes an A/B meaningful.

For the comparison, segment on the **post-failure window**, not the whole flight.
A whole-flight RMS averaged a 21% error down to 1.9% in this project; your recovery
transient is exactly the kind of short, localised event that averaging destroys.
No tool here does that split for you — `group_compare.py` aggregates the whole
flight — so window on the injection timestamp yourself before comparing.

And verify from the log that the rotor actually stopped, every single time. Two
runs in this project reported success while no failure was ever injected. Check
`actuator_motors.control[n]` goes to zero, or `motor_failure_mask != 0` — a command
being accepted is not proof it took effect.

---

## The three rules that will actually bite you

1. **`param save` + reboot after changing `MC_CTRL_ALG`**, and confirm the arm from
   `mc_controller status`. Both A/B arms once flew the same build unnoticed.
2. **A script printing "complete" is not evidence.** Check the ULog or a status
   topic. This produced four false-greens here, including a stick sweep that
   reported success while the mode never engaged.
3. **One SITL instance at a time.** Overlapping runs kill each other's simulator
   and produce empty or misleading logs. `pgrep -f bin/px4` before trusting
   anything.
