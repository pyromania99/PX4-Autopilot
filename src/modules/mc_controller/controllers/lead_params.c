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
 * Actuator lead compensation
 *
 * Multiplies the commanded body torque by (1 + MC_LEAD_TC * s), which cancels the rotor's
 * first-order pole and leaves the actuator looking like its bare transport delay. After
 * Lee et al., IEEE/ASME T-MECH 30(6) 5295-5306, 2025.
 *
 * 0 OFF  bit-identical to the controller before this existed. Default.
 * 1 ON   the feed-forward half of the source paper's loop shaping.
 *
 * WHY IT IS NOT THE SEAT. The seat rotates the wrench to cancel the phase a lag produces
 * under spin. It cannot recover the POLE's amplitude droop, 1/sqrt(1+(rT)^2). A lead
 * cancels the pole outright - phase and droop - and the two compose cleanly: the lead
 * handles the pole, which is invertible, and the seat handles the transport delay, which
 * is not. On this airframe at 20 rad/s the plant carries 43.8 deg of phase, 17.2 of it
 * delay and 26.6 of it pole, with 10.6 % droop on top.
 *
 * SECOND ORDER BENEFIT. The seat is measured to cancel 1.14-1.32 of the true phase when
 * the plant is a pure delay but only 0.39-0.63 when a pole is present. Removing the pole
 * therefore also moves the seat into the regime where it works.
 *
 * NOT the full source scheme. That also closes a PID loop on the torque error, which
 * needs the DELIVERED torque fed back - unavailable here without rotor RPM telemetry.
 * Only the feed-forward half is implemented, which is the half that carries the result:
 * the source's Fig. 4(a) reaches 1.5 Hz without the derivative term and 5.5 Hz with it.
 *
 * @value 0 Off
 * @value 1 On
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_INT32(MC_LEAD_EN, 0);

/**
 * Actuator pole being inverted
 *
 * The rotor/ESC first-order time constant T in Ga(s) = e^{-sL}/(1+sT). The lead is exactly
 * (1 + T s), so this parameter IS the inverse being applied and nothing else tunes it.
 *
 * MEASURE IT, do not guess. Cross-spectrum phase of vehicle_torque_setpoint against
 * vehicle_angular_velocity.xyz_derivative at the spin frequency, fitted as delay plus
 * pole - see notes/AIRFRAME_ID_METHOD.md. On this airframe ev_delay.ulg gives T = 25 ms
 * median with a 12.5-40 ms spread across windows, and that spread is the dominant
 * uncertainty in the whole scheme.
 *
 * SET IT TO THE BOTTOM OF YOUR MEASURED SPREAD, NOT THE MEDIAN. This is measured, not
 * reasoned. Level-3 hover, true pole 30 ms, believed pole swept with DMAX scaled by the
 * rule below so the feed-forward authority is held constant:
 *
 *   believed 12.5 ms  (-58%)   FLEW, indistinguishable from matched on every metric
 *   believed 20 ms    (-33%)   FLEW
 *   believed 30 ms    (matched) FLEW, n = 5
 *   believed 40 ms    (+33%)   MARGINAL - 2 crashes in 6, at 40 s and 44 s
 *   believed 50 ms    (+67%)   CRASH 2/2 at 18-21 s, 85% allocator saturation
 *
 * Under-inversion is free; over-inversion is not, and the asymmetry is NOT symmetric in
 * phase error: 20 and 40 ms are near-mirror errors (+12.5 deg against -12.1 deg) and 20
 * flies repeatably while 40 is a 1-in-3 coin flip. The penalty for being far too low is nil;
 * the penalty for being too high is a marginal aircraft at +33% and a lost one at +67%.
 *
 * AND THERE IS NO WARNING. The four surviving 40 ms runs hold 0.16-0.18 deg tilt for the
 * whole flight, indistinguishable from a matched TC; the two that died were equally flat
 * until one 5 s bin took them from 0.5 deg to 31 deg. Nothing separates the good runs from
 * the bad ones in advance. A setting that flies 4 times in 6 with no observable precursor
 * will pass a test flight and then kill the aircraft - which is worse than one that fails
 * outright. Do not tune this by flying it and seeing.
 *
 * 0 disables the lead regardless of MC_LEAD_EN.
 *
 * @min 0.0
 * @max 0.1
 * @unit s
 * @decimal 4
 * @increment 0.001
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_LEAD_TC, 0.020f);

/**
 * Lead differentiator bandwidth
 *
 * Sets both ESO gains: beta1 = 2*WN (critically damped) and beta2 = WN^2*sqrt(delta).
 *
 * DO NOT USE THE SOURCE PAPER'S TABLE I VALUES. Linearising its (19) inside the dead zone
 * gives a differentiator rolled off at wn = sqrt(beta2/sqrt(delta)) with damping
 * beta1/(2*wn); its listed beta1 = 1, beta2 = 500 puts damping at 0.004, an 89x resonant
 * peak. Driven against a known first-order plant reading 0.894/-24.5 deg uncompensated,
 * those gains give 1.410/-74.7 deg - worse than no compensation at all. Hence this
 * parameterisation.
 *
 * WHAT IT BUYS, measured on that plant (T = 25 ms, 20 rad/s spin, ideal 1.000/0 deg):
 *
 *   WN = 100 rad/s   0.981 / -25.8 deg    barely helps
 *   WN = 250         1.136 / -16.1 deg    recovers about a third of the pole phase
 *   WN = 400         1.106 /  -5.7 deg    recovers about three quarters
 *
 * WHAT IT COSTS. At 2 kHz with 1 % noise on the torque, the commanded sd rises 10 % at
 * WN = 100 and 27 % at WN = 250. There is no setting that both inverts cleanly and stays
 * quiet, so SWEEP THIS IN SIMULATION before flying it. The default is deliberately
 * conservative and is not a tuned value.
 *
 * @min 0.0
 * @max 1000.0
 * @unit rad/s
 * @decimal 1
 * @increment 10.0
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_LEAD_WN, 150.0f);

/**
 * Lead differentiator dead zone
 *
 * Half-width of fal()'s linear interior. Inside it the observer is linear, outside it the
 * gain is sub-linear (sqrt), which is what lets a large transient converge without a
 * correspondingly large small-signal gain.
 *
 * 0 means "use the timestep", which is what the source prescribes - but that ties the
 * nonlinearity to the loop rate and makes the effective bandwidth rate-dependent, which
 * is why the same WN measures differently at 250 Hz and 2 kHz. Setting it explicitly, to
 * roughly the torque noise floor, makes the tuning portable between rungs of the
 * simulation ladder. That portability is the whole point of the ladder.
 *
 * @min 0.0
 * @max 1.0
 * @unit Nm
 * @decimal 5
 * @increment 0.0001
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_LEAD_DLTA, 0.0f);

/**
 * Lead derivative clamp
 *
 * Hard limit on the estimated d(tau)/dt, in N m per second, before it is multiplied by
 * MC_LEAD_TC. (No @unit: PX4's whitelist has Nm but no Nm/s.)
 *
 * THIS PARAMETER CARRIES THE ENTIRE RESULT. It was first sized as TRQ_MAX/TC = 200, on
 * the reasoning that a derivative larger than "full torque within one TC" must be noise.
 * That rule is wrong, and measurably so: at 200 the level-3 rescue fails and at 25 it
 * flies, with every other setting held. Four values of MC_LEAD_TC (8, 15, 20, 30 ms) all
 * fly at DMAX = 25 and none fly at 200, so the mechanism is the clamp, not the amount of
 * inversion.
 *
 * WHY. TRQ_MAX/TC bounds the lead's own contribution at TC*DMAX = TRQ_MAX - it permits
 * the feed-forward term ALONE to command the full normalised torque, on top of whatever
 * the controller asked for. And MC_EIG_TRQ_MAX is only a normaliser: the allocator
 * saturates at about 0.50 of it. So the old default licensed a transient roughly 2x the
 * torque the airframe can deliver, from a differentiator, in one tick.
 *
 * SIZE IT INSTEAD as a fraction of the torque that can actually be DELIVERED:
 *
 *     DMAX ~= f * (0.50 * MC_EIG_TRQ_MAX) / MC_LEAD_TC,   f ~ 0.4
 *
 * On this airframe (TRQ_MAX 3.843, TC 0.030) that is 0.4*1.92/0.030 = 26, which is where
 * the flying value sits. The lead is a correction; it should not be able to out-command
 * the controller it corrects.
 *
 * The clamp is nonlinear, so this is not visible in the Bode plot of (1 + T s) - the
 * failure is unclamped observer spikes, not the linear lead. 0 disables the clamp, which
 * is not recommended on a vehicle.
 *
 * @min 0.0
 * @max 10000.0
 * @decimal 1
 * @increment 5.0
 * @group Multicopter Controller Framework
 */
PARAM_DEFINE_FLOAT(MC_LEAD_DMAX, 25.0f);
