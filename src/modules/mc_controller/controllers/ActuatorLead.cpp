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

#include "ActuatorLead.hpp"

#include <float.h>
#include <mathlib/mathlib.h>

using namespace matrix;

ActuatorLead::ActuatorLead(ModuleParams *parent) :
	ModuleParams(parent)
{
	ActuatorLead::updateParams();
	ActuatorLead::reset();
}

void ActuatorLead::updateParams()
{
	ModuleParams::updateParams();

	_enabled = (_param_mc_lead_en.get() != 0);
	_tc = math::max(_param_mc_lead_tc.get(), 0.f);
	_wn = math::max(_param_mc_lead_wn.get(), 0.f);
	_delta = math::max(_param_mc_lead_delta.get(), 0.f);
	_rate_max = math::max(_param_mc_lead_dmax.get(), 0.f);
}

void ActuatorLead::reset()
{
	_z1.setZero();
	_z2.setZero();
	_initialised = false;
}

float ActuatorLead::fal(float e, float alpha, float delta)
{
	/*
	 * Han's fal(): linear inside a dead zone of width delta, |e|^alpha outside. The
	 * linear interior is what stops the observer chattering when the error is already
	 * at the noise floor; the sub-linear exterior is what makes it converge fast on a
	 * large error without a correspondingly large gain. See the source paper's (20).
	 */
	if (delta <= FLT_EPSILON) {
		return e;
	}

	if (fabsf(e) <= delta) {
		return e / powf(delta, 1.f - alpha);
	}

	return sqrtf(fabsf(e)) * ((e < 0.f) ? -1.f : 1.f);	// alpha is fixed at 0.5
}

Vector3f ActuatorLead::apply(const Vector3f &torque, float dt)
{
	// Full bypass, and the default. Costs one compare when off.
	if (!_enabled || !(dt > 0.f) || !torque.isAllFinite()) {
		return torque;
	}

	// First tick: start the observer ON the signal rather than at zero, or the lead
	// emits a large spurious derivative while z1 catches up - on an armed vehicle that
	// is a torque spike, not a transient plot artefact.
	if (!_initialised) {
		_z1 = torque;
		_z2.setZero();
		_initialised = true;
		return torque;
	}

	/*
	 * ESO tracking differentiator, the source paper's (19):
	 *
	 *     e    = z1 - tau_des
	 *     z1'  = z2 - beta1 * e
	 *     z2'  = -beta2 * fal(e, 0.5, delta)
	 *
	 * Gains DERIVED from the bandwidth rather than taken from the source's Table I,
	 * which is nearly undamped - see the header. Linearised inside the dead zone this
	 * is a differentiator rolled off at wn with damping beta1/(2 wn), so
	 *
	 *     beta1 = 2 wn        (zeta = 1)
	 *     beta2 = wn^2 sqrt(delta)
	 */
	const float delta = (_delta > FLT_EPSILON) ? _delta : dt;
	const float beta1 = 2.f * _wn;
	const float beta2 = _wn * _wn * sqrtf(delta);

	for (int i = 0; i < 3; i++) {
		const float e = _z1(i) - torque(i);
		const float z1_dot = _z2(i) - beta1 * e;
		const float z2_dot = -beta2 * fal(e, 0.5f, delta);

		_z1(i) += dt * z1_dot;
		_z2(i) += dt * z2_dot;

		if (!PX4_ISFINITE(_z1(i)) || !PX4_ISFINITE(_z2(i))) {
			_z1(i) = torque(i);
			_z2(i) = 0.f;
		}

		// A derivative clamp is not cosmetic. (1 + T s) has gain 15.7x at 100 Hz, so an
		// unbounded derivative turns a gyro spike into a motor spike. MC_LEAD_DMAX is
		// the one guard between the differentiator and the mixer.
		if (_rate_max > FLT_EPSILON) {
			_z2(i) = math::constrain(_z2(i), -_rate_max, _rate_max);
		}
	}

	// The whole compensator: u = tau + T * d(tau)/dt. Yaw is included - the pole is a
	// property of the rotors, not of the axis, and yaw torque goes through the same
	// first-order lag.
	return torque + _z2 * _tc;
}
