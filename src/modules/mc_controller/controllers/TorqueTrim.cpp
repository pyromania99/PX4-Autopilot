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

#include "TorqueTrim.hpp"

#include <mathlib/mathlib.h>

using namespace matrix;

/// Longer than any real gyro interval; a bigger dt is a scheduling stall, not an interval
/// the law observed, and integrating across it would step the trim.
static constexpr float kMaxDt = 0.1f;

TorqueTrim::TorqueTrim(ModuleParams *parent) :
	ModuleParams(parent)
{
	TorqueTrim::updateParams();
}

void TorqueTrim::updateParams()
{
	ModuleParams::updateParams();

	_gain = math::max(_param_mc_att_trim_i.get(), 0.f);
	_limit = math::max(_param_mc_att_trim_lim.get(), 0.f);

	// Lowering the limit in flight must take effect now, not when the trim next moves.
	_trim(0) = math::constrain(_trim(0), -_limit, _limit);
	_trim(1) = math::constrain(_trim(1), -_limit, _limit);
}

Vector3f TorqueTrim::apply(const Vector3f &torque_law, float dt)
{
	Vector3f torque = torque_law;

	if (!enabled()) {
		_trim.setZero();
		return torque;
	}

	const bool dt_valid = PX4_ISFINITE(dt) && (dt > 0.f) && (dt < kMaxDt);

	for (int i = 0; i < 2; i++) {
		if (!PX4_ISFINITE(torque_law(i))) {
			// Leave the axis alone: the law's NaN is the framework's to catch, and a NaN
			// must never reach the integrator.
			continue;
		}

		// Anti-windup, the same conditional integration as the trajectory stage: stop
		// integrating INTO a limit, keep integrating out of it. The trim moves in the
		// direction of the law's torque, so it deepens the output clamp exactly when the
		// law's torque and the total already point the same way past +-1.
		const float total = torque_law(i) + _trim(i);
		const bool into_clamp = (fabsf(total) >= 1.f) && (total * torque_law(i) > 0.f);

		if (dt_valid && !_allocator_saturated && !into_clamp) {
			_trim(i) += _gain * torque_law(i) * dt;
			_trim(i) = math::constrain(_trim(i), -_limit, _limit);
		}

		torque(i) = torque_law(i) + _trim(i);
	}

	return torque;
}
