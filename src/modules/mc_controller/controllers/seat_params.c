/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/**
 * @file seat_params.c
 * Command-frame rotation of the roll/pitch torque, which cancels the phase rotation
 * actuator lag applies under a continuous spin. See Seat.hpp for the mechanism.
 *
 * MC_SEAT_MODE = 0 is a full bypass and is the default: nothing here changes the
 * vehicle until it is deliberately turned on.
 *
 * Only MC_CTRL_ALG=3 (the eigen law) carries a seat. The observable needs the law's own
 * commanded angular acceleration and its assumed plant drift, both of which are internal
 * to that controller - so this is not a framework-level feature and the other laws ignore
 * these parameters entirely.
 *
 * WHERE IT MATTERS. Validated at level 1 (Pegasus + Python, examples/utils/seat.py) at
 * r = 29.5 rad/s with 20 ms of rotor lag: the eigen law lost control at t = 13.7 s
 * without it and flew 120 s with it, at 0.19 deg roll/pitch RMS. Below ~20 rad/s it
 * changes little either way, because there the lag does not threaten the vehicle.
 */

/**
 * Seat mode
 *
 * Rotates the commanded roll/pitch torque to cancel the rotation actuator lag applies to
 * it while the vehicle spins. 0 bypasses the feature entirely.
 *
 * Mode 1 computes the angle from MC_SEAT_TAU_A and MC_SEAT_T, which have to be right for
 * it to help. It is a NON-ADAPTIVE REFERENCE POINT, not a fallback: at level 1 it was
 * worse than no seat at all (crashed at 6.0 s against the baseline's 13.7 s), because the
 * open-loop model angle is not what the closed loop actually wants.
 *
 * Mode 2 measures the misalignment in flight and needs no lag estimate. It is the mode
 * that works.
 *
 * Roll and pitch only. Yaw torque is never rotated.
 *
 * @value 0 Disabled (bypass)
 * @value 1 Fixed model rotation
 * @value 2 Adaptive rotation
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_INT32(MC_SEAT_MODE, 0);

/**
 * Seat update law
 *
 * How the adaptation is driven. Both share the same fixed point, so this changes
 * robustness rather than where the angle ends up.
 *
 * 0 drives on the signed angle from atan2, which normalises the signal magnitudes away -
 * so every tick takes a full-size step regardless of how little signal produced it.
 *
 * 1 drives on the raw cross product, which is that angle weighted by the signal
 * magnitudes: a starved channel then suppresses its own update with no threshold to tune,
 * and there is no atan2 near zero to guard. At level 1 this cut the angle's wander 3x in
 * hover. It is the default.
 *
 * NOTE MC_SEAT_K IS DIMENSIONAL FOR LAW 1: it absorbs the units of the two accelerations,
 * so the effective bandwidth scales with signal amplitude and the gain does not carry
 * across operating points the way the angle law's does.
 *
 * @value 0 Angle (atan2)
 * @value 1 Cross product
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_INT32(MC_SEAT_LAW, 1);

/**
 * Seat adaptation gain
 *
 * Rate at which the rotation angle chases the measured misalignment. Mode 2 only.
 *
 * For MC_SEAT_LAW=0 this is rad/s of angle per rad of misalignment and the misalignment
 * decays as exp(-K t), so it is directly the adaptation bandwidth. For MC_SEAT_LAW=1 (the
 * default law) it additionally absorbs |xd||xa|, the product of the two angular
 * acceleration magnitudes, and is therefore NOT the same number as MC_SEAT_LAW=0's - using
 * the ANGLE-law's k=2.0 under CROSS drove theta_s across its whole range in a single 4 ms
 * tick (measured 2026-09-11: |xd||xa| averages ~140 under real actuator lag, so k=2.0
 * gives a ~280 rad/s drive). Measured 2026-09-12 on this airframe with the corrected xd/xa
 * pairing: |xd||xa| averages ~10-30 in level flight, so k=0.02-0.05 gives a 0.5-1.5 rad/s
 * adaptation bandwidth - the default below sits at the low end of that, and should be
 * rechecked if the airframe's inertia, authority, or torque scale differ.
 *
 * It has to sit in a window either way: fast enough that the steady tracking lag stays
 * inside the angle tolerance, and far slower than the spin rate r so the adaptation never
 * interacts with the mode it is compensating.
 *
 * @min 0.0
 * @max 20.0
 * @decimal 3
 * @increment 0.01
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_SEAT_K, 0.02f);

/**
 * Seat angle limit
 *
 * Maximum magnitude of the rotation. Hard-ceilinged at 1.5 rad in code whatever is set.
 *
 * The limit is a realizability one, not a safety margin: holding an angle theta costs a
 * pole magnitude growing as 1/cos(theta), so pi/2 is where the required authority
 * diverges. A converged angle sitting AT this limit means the seat has run out of range
 * and the lag is no longer being cancelled - that is a signal to look at the lag, not to
 * raise the limit. At level 1 the converged value was 0.15 rad at r = 29.5, so the
 * default leaves a wide margin.
 *
 * @min 0.0
 * @max 1.5
 * @unit rad
 * @decimal 2
 * @increment 0.05
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_SEAT_THMAX, 1.4f);

/**
 * Seat spin-rate gate
 *
 * Adaptation runs only while the measured yaw rate exceeds this. Below it the delay
 * applies no rotation to cancel, so the measured misalignment is noise and the angle is
 * held rather than driven.
 *
 * Set it clear of the yaw rates ordinary flight reaches and well under the spin that
 * follows a rotor failure - around 13 rad/s on hardware, 30 rad/s in simulation.
 *
 * @min 0.0
 * @max 30.0
 * @unit rad/s
 * @decimal 1
 * @increment 0.5
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_SEAT_RMIN, 5.0f);

/**
 * Seat adaptation sign
 *
 * Direction the angle moves for a given measured misalignment.
 *
 * +1 is the sign the derivation gives, and it does NOT depend on which way the vehicle
 * spins - the spin rate already enters through the measured misalignment. It is exposed
 * because the derivation cannot settle the sign conventions of the mixer geometry and of
 * the measured angular acceleration, and getting it backwards is the one silent failure
 * of this feature: the angle runs the wrong way and DOUBLES the misalignment instead of
 * cancelling it. At level 1, -1 drove the angle to its rail and crashed the vehicle.
 *
 * Confirm it from a log before trusting it: mc_controller_status.seat_theta must settle,
 * and seat_alpha must not sit at pi/2.
 *
 * @value -1 Negative
 * @value 1 Positive
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_INT32(MC_SEAT_KSIGN, 1);

/**
 * Seat transport delay
 *
 * Pure transport delay from a published torque setpoint to the rotors acting on it -
 * mixer, output driver, ESC command latency. MODE 1 ONLY; mode 2 never reads it.
 *
 * At spin rate r this contributes exactly r * MC_SEAT_TAU_A to the model angle, with no
 * droop, because a pure delay is all-pass.
 *
 * Measured 2026-09-12 on this airframe at rotor_tau=0 (so isolating whatever delay PX4's
 * own pipeline has, independent of any modelled rotor lag): with the seat inert, the
 * corrected xd/xa pairing reads a systematic misalignment of -0.84 rad at r=27.8, and
 * |xa|/|xd| = 0.933 - close enough to the all-pass value of 1.0 (a first-order lag
 * producing this much phase would droop to 0.66) that this is delay, not lag. That is
 * ~30 ms of transport delay baked into PX4's own loop/mixer/allocator/uORB pipeline that
 * the previous default of 2 ms did not represent at all.
 *
 * @min 0.0
 * @max 0.05
 * @unit s
 * @decimal 4
 * @increment 0.0005
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_SEAT_TAU_A, 0.030f);

/**
 * Seat actuator lag
 *
 * First-order lag of the rotor and ESC responding to a commanded thrust change. MODE 1
 * ONLY; mode 2 never reads it.
 *
 * Unlike a transport delay this one droops as well as rotates: it contributes
 * atan(r * MC_SEAT_T) of angle and 1/sqrt(1+(r*T)^2) of magnitude, and the seat cannot
 * recover the magnitude - 7% is lost at r = 20 with T = 20 ms, 30% at r = 30.
 *
 * The default matches ROTOR_TAU_PLACEHOLDER in the Pegasus simulator's
 * quadratic_thrust_curve.py, which is itself a PLACEHOLDER rather than a measurement for
 * this airframe.
 *
 * @min 0.0
 * @max 0.2
 * @unit s
 * @decimal 4
 * @increment 0.001
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_SEAT_T, 0.020f);
