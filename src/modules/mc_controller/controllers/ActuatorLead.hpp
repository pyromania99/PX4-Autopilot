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
 * @file ActuatorLead.hpp
 * @brief Inverts the rotor's first-order pole so the actuator behaves as a faster one.
 *        MC_LEAD_*. After Lee et al., IEEE/ASME T-MECH 30(6) 5295-5306, 2025.
 *
 * WHAT IT IS, AND WHY IT IS NOT THE SEAT
 * The actuator between a commanded torque and a delivered one is roughly
 *
 *     Ga(s) = e^{-s L} / (1 + s T)        L transport delay, T rotor/ESC pole
 *
 * and under a spin at rate r both terms rotate the delivered wrench in the body frame.
 * The seat cancels that rotation geometrically. It cannot do anything else, and in
 * particular it cannot recover the POLE's amplitude droop 1/sqrt(1+(rT)^2).
 *
 * This does the other half. Modelling the rotor as tau_dot = (u - tau)/T, the exact
 * inverse of that pole is
 *
 *     u = tau_des + T * d(tau_des)/dt          i.e. multiply by (1 + T s)
 *
 * which cancels the pole ENTIRELY - phase and droop - leaving the actuator looking like
 * the bare delay e^{-s L}. A delay is non-minimum phase and cannot be inverted by
 * anything causal, so what is left is exactly the seat's job. The two compose:
 *
 *     lead handles the pole          (invertible, needs T)
 *     seat handles the delay         (not invertible, needs no model)
 *
 * MEASURED ON THIS AIRFRAME (notes/AIRFRAME_ID_METHOD.md, ev_delay.ulg): at 20 rad/s the
 * plant carries 43.8 deg of phase, of which 17.2 deg is delay and 26.6 deg is the pole,
 * and the pole also droops 10.6 %. Inverting the pole leaves 17.2 deg for the seat - and
 * the seat handles a PURE DELAY well (measured ratio 1.14-1.32 of the true phase) where
 * it stalls at 0.39-0.63 when a pole is present. So this is expected to help twice: it
 * removes the pole, and it moves the seat into the regime where it works.
 *
 * THE SOURCE'S PUBLISHED GAINS DO NOT WORK AS STATED. Linearising its (19) inside the
 * dead zone gives z2/u = b2 k s / (s^2 + b1 s + b2 k) with k = 1/sqrt(delta): a
 * differentiator rolled off at wn = sqrt(b2 k) with damping zeta = b1/(2 wn). Its Table I
 * values (b1 = 1, b2 = 500) put zeta at 0.004 - an 89x to 126x resonant peak at 14-20 Hz.
 * Driven with a known first-order plant that reads 0.894 / -24.5 deg uncompensated, those
 * gains produce 1.410 / -74.7 deg: far WORSE than no compensation. So this implementation
 * is parameterised by the differentiator BANDWIDTH instead, deriving
 *
 *     b1 = 2 wn   (zeta = 1, critically damped)      b2 = wn^2 sqrt(delta)
 *
 * which is monotonic and verifiable. MC_LEAD_WN is the knob.
 *
 * WHAT IS ACHIEVABLE, measured against that same plant (T = 25 ms, 20 rad/s, target
 * 1.000 / 0.00, uncompensated 0.894 / -24.5 deg):
 *
 *     wn 100 rad/s   0.981 / -25.8      barely helps
 *     wn 250         1.136 / -16.1      recovers a third of the pole phase
 *     wn 400         1.106 /  -5.7      recovers three quarters
 *
 * The inversion is PARTIAL and bounded by wn, which is bounded by noise: at 2 kHz with
 * 1 % torque noise the commanded sd rises 10 % at wn = 100 and 27 % at wn = 250. There is
 * no setting that both inverts cleanly and stays quiet, so this needs a sweep in
 * simulation before it flies. Do not assume the defaults are tuned.
 *
 * WHY A TRACKING DIFFERENTIATOR AND NOT A DIFFERENCE
 * (1 + T s) has gain |1 + i w T|: 1.1x at a 20 rad/s spin, but 7.9x at 50 Hz and 15.7x at
 * 100 Hz. A naive difference of tau_des would inject that much gyro noise straight into
 * the motors. The ESO-based tracking differentiator of the source paper (its Algorithm 1)
 * estimates the derivative while filtering, and is the reason the derivative is usable at
 * all. It is six lines and about 830 ns per cycle on this MCU family.
 *
 * WHAT IS NOT IMPLEMENTED. The source paper also closes a PID loop on the torque ERROR
 * tau_des - tau_B (its eq. 17), which needs the DELIVERED torque tau_B fed back from the
 * vehicle. Without rotor RPM telemetry (DSHOT_BIDIR_EN) the only route to tau_B here is
 * omega_dot minus the modelled drift, and that reconstruction is already known to be
 * unreliable under spin - it is what refuted the seat's magnitude term, reading 0.044 to
 * 1.225 where the truth was 1.000. So only the feed-forward half is implemented. That is
 * the half that carries the result: the source paper's Fig. 4(a) has their loop shaping
 * WITHOUT the derivative term reaching 1.5 Hz and WITH it reaching 5.5 Hz.
 *
 * ORDER. Apply this LAST, after the seat, immediately before normalisation - the actuator
 * receives lead(seat(tau)), so it delivers e^{-sL} * seat(tau). The seat's own observable
 * is unaffected: it forms xd from the pre-seat torque and xa from the measurement, so it
 * simply sees a faster actuator and adapts to the smaller residual. Nothing about this
 * competes with the adaptation.
 */

#pragma once

#include <matrix/matrix/math.hpp>
#include <px4_platform_common/module_params.h>

class ActuatorLead : public ModuleParams
{
public:
	explicit ActuatorLead(ModuleParams *parent);
	~ActuatorLead() override = default;

	ActuatorLead(const ActuatorLead &) = delete;
	ActuatorLead &operator=(const ActuatorLead &) = delete;

	/// Drop the differentiator state. Call wherever the owning controller resets.
	void reset();

	/**
	 * u = tau + T * d(tau)/dt, with the derivative from an ESO tracking differentiator.
	 *
	 * Returns @p torque unchanged when MC_LEAD_EN is 0, which is the default, so the
	 * controller is bit-identical to before this existed until it is switched on.
	 *
	 * Real-time safe: no allocation, no blocking, one sqrt per axis.
	 *
	 * @param torque body torque [N m], already rotated by the seat
	 * @param dt     [s] control timestep
	 */
	matrix::Vector3f apply(const matrix::Vector3f &torque, float dt);

	/// Estimated d(tau)/dt [N m/s] on the last tick, per axis. Diagnostics only.
	const matrix::Vector3f &derivative() const { return _z2; }

	/// Cached MC_LEAD_EN, so a caller can skip the work entirely when disabled.
	bool enabled() const { return _enabled; }

protected:
	void updateParams() override;

private:
	/// Han's nonlinear gain: linear inside the dead zone, square-root outside.
	static float fal(float e, float alpha, float delta);

	matrix::Vector3f _z1{};	///< tracked torque
	matrix::Vector3f _z2{};	///< tracked derivative - this is the output
	bool _initialised{false};

	bool _enabled{false};
	float _tc{0.f};		///< [s] MC_LEAD_TC, the pole being inverted
	float _wn{0.f};		///< [rad/s] MC_LEAD_WN, differentiator bandwidth - the real knob
	float _delta{0.f};	///< MC_LEAD_DELTA, fal() dead-zone half-width; 0 means "use dt"
	float _rate_max{0.f};	///< [N m/s] MC_LEAD_DMAX, clamp on the derivative

	DEFINE_PARAMETERS(
		(ParamInt<px4::params::MC_LEAD_EN>)     _param_mc_lead_en,
		(ParamFloat<px4::params::MC_LEAD_TC>)   _param_mc_lead_tc,
		(ParamFloat<px4::params::MC_LEAD_WN>)   _param_mc_lead_wn,
		(ParamFloat<px4::params::MC_LEAD_DLTA>) _param_mc_lead_delta,
		(ParamFloat<px4::params::MC_LEAD_DMAX>) _param_mc_lead_dmax
	)
};
