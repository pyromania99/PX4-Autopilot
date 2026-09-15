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
 * Adaptive pole angle enable
 *
 * Adapts theta_b, an OFFSET added to the eigen law's designed pole angle
 * atan2(MC_EIG_B, MC_EIG_WN). Independent of the seat (MC_SEAT_*): the seat rotates the
 * commanded wrench and changes no design parameter, this changes the design and rotates
 * nothing. They can run together or separately.
 *
 * At 0 the controller is bit-identical to an unmodified one - theta_b is an offset, so
 * disabling it is exactly "offset zero", not a different code path.
 *
 * KNOWN RESULT, level-1 harness: the observable this drives on (the angle from the
 * acceleration M_dyn asks for to the one achieved) has NO interior fixed point. |alpha_b|
 * stayed >= 1.12 rad however slowly theta_b was moved, so theta_b runs monotonically to
 * MC_POLE_MAX and slowing it only reduces the damage. Enable this expecting to measure
 * that, not to benefit from it.
 *
 * 0 disabled. 1 = brief v3 section 4's gradient law, kept ONLY so the broken observable
 * stays measurable against the replacement - it ramps theta_b to MC_POLE_MAX and crashed
 * the vehicle in 3-9 s at level 2. 2 = extremum seeking on the lateral tracking cost, with
 * an exogenous reference and a DITHER-MEASURED gradient sign.
 *
 * @min 0
 * @max 2
 * @value 0 Disabled
 * @value 1 Gradient (brief v3 s4) - known ill-formed, comparison only
 * @value 2 Extremum seeking on lateral tracking
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_INT32(MC_POLE_EN, 0);

/**
 * Adaptive pole angle gain
 *
 * Rate at which the pole angle offset chases the measured misalignment, in rad/s of angle
 * per rad of misalignment. The ANGLE law (atan2) is used, not the seat's CROSS law, so
 * this gain is dimensionless in the same sense MC_SEAT_K is at MC_SEAT_LAW=0 - it is
 * directly the adaptation bandwidth and does not absorb the signal magnitudes.
 *
 * Must sit far below the spin rate r, and below the seat's bandwidth when both run: the
 * level-1 brief calls for k_s/k_b >= 10 so the two adaptations do not interact. The
 * level-1 default is 0.2.
 *
 * @min 0.0
 * @max 5.0
 * @decimal 3
 * @increment 0.01
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_POLE_K, 0.2f);

/**
 * Adaptive pole angle limit
 *
 * Largest offset |theta_b| the adaptation may reach [rad]. Clipped internally to 1.5 rad
 * regardless: holding a pole angle theta costs a pole magnitude growing as 1/cos(theta),
 * so approaching pi/2 is a realizability limit, not a safety margin.
 *
 * @min 0.0
 * @max 1.5
 * @unit rad
 * @decimal 2
 * @increment 0.05
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_POLE_MAX, 1.4f);

/**
 * Adaptive pole angle minimum spin rate
 *
 * Below this |yaw rate| the adaptation is frozen [rad/s]. The misalignment this drives on
 * is only defined against a spin frequency; with no spin, whatever the angle reads is
 * noise and integrating it is a random walk.
 *
 * @min 0.0
 * @max 50.0
 * @unit rad/s
 * @decimal 1
 * @increment 0.5
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_POLE_RMIN, 5.0f);

/**
 * Adaptive pole angle dither amplitude
 *
 * MC_POLE_EN=2 only. Amplitude of the probing oscillation added to theta_b [rad]. This is
 * what makes the gradient measurable: without it there is nothing to correlate the cost
 * against and the sign of d(cost)/d(theta_b) is unknown.
 *
 * Has to be large enough to move a SHALLOW cost. The archived static-b sweep spans only
 * 0.7-3 percent of cost across the whole b range, so too small a dither buys a gradient
 * estimate indistinguishable from noise; too large and the probing itself degrades flight.
 *
 * @min 0.0
 * @max 0.5
 * @unit rad
 * @decimal 3
 * @increment 0.01
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_POLE_DITH, 0.12f);

/**
 * Adaptive pole angle dither frequency
 *
 * MC_POLE_EN=2 only. Frequency of the probing oscillation [Hz]. Needs a three-way timescale
 * separation to work: well BELOW the spin rate r (so the probe does not interact with the
 * mode being compensated), and well ABOVE the MC_POLE_KES descent rate (so the cost has
 * settled before the estimate moves). At r = 29.5 rad/s (4.7 Hz) roughly 0.3-0.5 Hz.
 *
 * @min 0.05
 * @max 3.0
 * @unit Hz
 * @decimal 2
 * @increment 0.05
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_POLE_FDIT, 0.4f);

/**
 * Adaptive pole angle descent gain
 *
 * MC_POLE_EN=2 only. Rate at which the estimate descends the measured gradient. Must be
 * the slowest thing in the loop - below the dither frequency, which is itself well below
 * the spin rate. Deliberately slow: a shallow cost bowl means the gradient estimate is
 * noisy, and the cure for that is averaging, not gain.
 *
 * @min 0.0
 * @max 20.0
 * @decimal 2
 * @increment 0.1
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_POLE_KES, 2.0f);

/**
 * Adaptive pole angle minimum commanded acceleration
 *
 * MC_POLE_EN=2 only. Below this commanded (or achieved) horizontal acceleration the update
 * is frozen [m/s^2]. The cost is the ANGLE between two horizontal acceleration vectors, and
 * the angle between two near-zero vectors is noise - so in a pure position hold, where
 * there is no lateral demand at all, this law correctly does nothing. It needs the vehicle
 * to actually be translating to have anything to measure.
 *
 * @min 0.0
 * @max 5.0
 * @unit m/s^2
 * @decimal 2
 * @increment 0.05
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_POLE_AMIN, 0.3f);
