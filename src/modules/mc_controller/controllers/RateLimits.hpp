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
 * @file RateLimits.hpp
 * @brief The attitude loop's rate-setpoint ceiling, MC_ROLLRATE_MAX / MC_PITCHRATE_MAX /
 *        MC_YAWRATE_MAX.
 *
 * WHAT IT IS. Stock applies these at the very end of AttitudeControl::update()
 * (AttitudeControl.cpp:110-113), after the attitude error has been turned into rates and
 * after the yaw feed-forward has been added. No law in this module applied them at all, so
 * a large attitude error - a mode change onto a distant setpoint, a stick slammed to the
 * corner, a setpoint step out of a flight task - produced a rate demand bounded by nothing
 * but the angle gain, and the inner loop then chased it with whatever torque that implies.
 *
 * WHERE IT DOES NOT APPLY, also following stock: Acro. There the rate setpoint IS the
 * pilot's command and is already bounded by MC_ACRO_*_MAX upstream in mc_manual_mapping;
 * stock's rate controller never sees MC_*RATE_MAX either. A law therefore applies this to
 * what its ANGLE stage produced, not to command.rate_sp.
 *
 * COMPOSITION, like TrajectoryStage and Seat: a law opts in by holding one and calling it.
 * A ModuleParams child, so its gains refresh through the same cascade the owning law does.
 */

#pragma once

#include <matrix/matrix/math.hpp>
#include <px4_platform_common/module_params.h>

class RateLimits : public ModuleParams
{
public:
	explicit RateLimits(ModuleParams *parent);
	~RateLimits() override = default;

	RateLimits(const RateLimits &) = delete;
	RateLimits &operator=(const RateLimits &) = delete;

	/**
	 * Clamp a rate setpoint per axis to the configured maxima.
	 *
	 * A non-finite axis is passed through untouched: deciding what a NaN rate setpoint
	 * means belongs to the law that produced it, and the framework's output check catches
	 * it either way.
	 */
	matrix::Vector3f apply(const matrix::Vector3f &rate_setpoint) const;

	/// [rad/s] per axis, for tests and diagnostics.
	const matrix::Vector3f &limit() const { return _limit; }

protected:
	void updateParams() override;

private:
	matrix::Vector3f _limit{};	///< [rad/s]

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::MC_ROLLRATE_MAX>)  _param_mc_rollrate_max,
		(ParamFloat<px4::params::MC_PITCHRATE_MAX>) _param_mc_pitchrate_max,
		(ParamFloat<px4::params::MC_YAWRATE_MAX>)   _param_mc_yawrate_max
	)
};
