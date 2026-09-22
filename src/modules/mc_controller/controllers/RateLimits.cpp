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

#include "RateLimits.hpp"

#include <mathlib/mathlib.h>

using namespace matrix;

RateLimits::RateLimits(ModuleParams *parent) :
	ModuleParams(parent)
{
	RateLimits::updateParams();
}

void RateLimits::updateParams()
{
	ModuleParams::updateParams();

	// Cached in radians so apply() runs at gyro rate without touching the parameter system
	// or converting. Floored at zero rather than at some small positive value: a zero
	// maximum is a legitimate "do not command rotation on this axis", and it is what
	// MC_YAWRATE_MAX=0 already means to stock.
	_limit = Vector3f(math::radians(math::max(_param_mc_rollrate_max.get(), 0.f)),
			  math::radians(math::max(_param_mc_pitchrate_max.get(), 0.f)),
			  math::radians(math::max(_param_mc_yawrate_max.get(), 0.f)));
}

Vector3f RateLimits::apply(const Vector3f &rate_setpoint) const
{
	Vector3f limited = rate_setpoint;

	for (int i = 0; i < 3; i++) {
		if (PX4_ISFINITE(limited(i))) {
			limited(i) = math::constrain(limited(i), -_limit(i), _limit(i));
		}
	}

	return limited;
}
