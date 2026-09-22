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

#include "Seat.hpp"

#include <float.h>
#include <mathlib/mathlib.h>

using namespace matrix;

namespace
{
/**
 * Smallest vector magnitude with a usable direction, for the ANGLE law only.
 * atan2 of a near-zero pair returns 0 rather than NaN - a confident wrong answer
 * feeding an integrator - so it has to be gated. The CROSS law needs no such gate:
 * its drive goes to zero with the magnitudes by construction, which is the point.
 */
constexpr float kEpsAngle = 1e-4f;
} // namespace

Seat::Seat(ModuleParams *parent) :
	ModuleParams(parent)
{
	Seat::updateParams();
	Seat::reset();
}

void Seat::updateParams()
{
	ModuleParams::updateParams();

	_mode = static_cast<Mode>(_param_mc_seat_mode.get());
	_law = static_cast<Law>(_param_mc_seat_law.get());

	_k = math::max(_param_mc_seat_k.get(), 0.f);
	_theta_max = math::constrain(_param_mc_seat_thmax.get(), 0.f, kThetaCeiling);
	_r_min = math::max(_param_mc_seat_rmin.get(), 0.f);
	_tau_a = math::max(_param_mc_seat_tau_a.get(), 0.f);
	_T = math::max(_param_mc_seat_t.get(), 0.f);
	_sat_gate = (_param_mc_seat_satgate.get() != 0);

	// Only the sign is used; the gain's magnitude lives in MC_SEAT_K.
	_k_sign = (_param_mc_seat_ksign.get() < 0) ? -1.f : 1.f;

	// A parameter change that shrinks THMAX has to pull the live angle in with it, or
	// theta_s sits outside its own limit until the next adapting tick.
	_theta_s = math::constrain(_theta_s, -_theta_max, _theta_max);
}

void Seat::reset()
{
	_theta_s = 0.f;
	_alpha = NAN;
}

float Seat::psiModel(float r) const
{
	// r*tau_a is a transport delay's exact rotation (a pure delay is all-pass);
	// atan(r*T) is the first-order lag's phase. Signed in r throughout.
	return r * _tau_a + atanf(r * _T);
}

Vector3f Seat::apply(const Vector3f &torque, const Vector2f &xd, const Vector2f &xa,
		     float spin_rate, float dt)
{
	// Full bypass, and the default. Costs one compare when off.
	if (_mode == Mode::Off) {
		return torque;
	}

	if (_mode == Mode::Fixed) {
		// Deterministic, recomputed from the live spin rate every tick so it tracks
		// r with no state at all. The non-adaptive baseline to compare against - and
		// note it was WORSE than no seat at level 1 (crashed at 6.0 s against the
		// baseline's 13.7 s), so it is a reference point, not a fallback.
		_theta_s = math::constrain(_k_sign * psiModel(spin_rate), -_theta_max, _theta_max);
		_alpha = NAN;

	} else {
		adapt(xd, xa, spin_rate, dt);
	}

	// The whole compensator: a static 2x2 rotation. No dt, no division, no state.
	const float c = cosf(_theta_s);
	const float s = sinf(_theta_s);

	Vector3f out = torque;
	out(0) = c * torque(0) - s * torque(1);
	out(1) = s * torque(0) + c * torque(1);
	return out;	// out(2), yaw, is never rotated
}

float Seat::measure(const Vector2f &xd, const Vector2f &xa)
{
	if (!xd.isAllFinite() || !xa.isAllFinite()
	    || (xd.norm() <= kEpsAngle) || (xa.norm() <= kEpsAngle)) {
		return NAN;
	}

	// cross = |xd||xa| sin(alpha), dot = |xd||xa| cos(alpha).
	return atan2f(xd(0) * xa(1) - xd(1) * xa(0),
		      xd(0) * xa(0) + xd(1) * xa(1));
}

void Seat::adapt(const Vector2f &xd, const Vector2f &xa, float r, float dt)
{
	_alpha = NAN;

	// Misalignment is unobservable when the vehicle is not spinning: there is no spin
	// frequency for the delay to rotate about, so whatever the angle reads is noise.
	//
	// The saturation term is behind MC_SEAT_SATGATE (default 1, the behaviour every
	// archived run was flown with). See setSaturated() for what it costs and why it is
	// still the default.
	if ((_sat_gate && _saturated) || (fabsf(r) <= _r_min) || !(dt > 0.f)
	    || !xd.isAllFinite() || !xa.isAllFinite()) {
		return;
	}

	// cross = |xd||xa| sin(alpha); the cos partner lives in measure(), which is where
	// the angle is now formed.
	const float cross = xd(0) * xa(1) - xd(1) * xa(0);

	// Recorded for BOTH laws, because it is the diagnostic that says whether the
	// channel carries any direction at all: |alpha| sitting at pi/2 means the pair is
	// uncorrelated and the seat is running on noise. It only DRIVES the angle law.
	_alpha = measure(xd, xa);
	const bool measurable = PX4_ISFINITE(_alpha);

	float step;

	if (_law == Law::Cross) {
		step = cross;

	} else {
		if (!measurable) {
			return;
		}

		step = _alpha;
	}

	if (!PX4_ISFINITE(step)) {
		_alpha = NAN;
		return;
	}

	/*
	 * SIGN. With alpha the angle FROM the commanded direction TO the measured one, the
	 * delivered direction is R(theta_s - psi)*xd, so alpha = theta_s - psi and
	 *
	 *     d theta_s/dt = -k*alpha  =>  theta_s -> psi.
	 *
	 * This does NOT depend on the sign of r: r already enters through the measured
	 * alpha. MC_SEAT_KSIGN = +1 is the derived value and was confirmed at level 1
	 * (ksign = -1 there drove the angle to its rail and crashed the vehicle); it exists
	 * because the derivation cannot settle the sign conventions of the mixer geometry
	 * and the measured angular acceleration, and getting it backwards DOUBLES the
	 * misalignment instead of cancelling it.
	 */
	_theta_s = math::constrain(_theta_s - _k_sign * _k * step * dt, -_theta_max, _theta_max);
}
