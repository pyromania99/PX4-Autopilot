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
 * @file StickToRateSetpoint.hpp
 * @brief Map manual stick input to a body rate setpoint (Acro mode).
 *
 * Extracted verbatim from MulticopterRateControl::Run() so that both the stock
 * mc_rate_control module and the pluggable controller framework share one
 * implementation of the acro stick mapping.
 */

#pragma once

#include <matrix/matrix/math.hpp>
#include <px4_platform_common/module_params.h>

#include <uORB/topics/manual_control_setpoint.h>

class StickToRateSetpoint : public ModuleParams
{
public:
	explicit StickToRateSetpoint(ModuleParams *parent);
	~StickToRateSetpoint() override = default;

	/**
	 * Generate a body rate setpoint and a body thrust setpoint from stick input.
	 */
	void update(const manual_control_setpoint_s &manual_control_setpoint, matrix::Vector3f &rate_setpoint,
		    matrix::Vector3f &thrust_setpoint) const;

	/**
	 * Maximum acro rates [rad/s], derived from MC_ACRO_{R,P,Y}_MAX.
	 */
	const matrix::Vector3f &acroRateMax() const { return _acro_rate_max; }

protected:
	void updateParams() override;

private:
	matrix::Vector3f _acro_rate_max;	///< max attitude rates in acro mode [rad/s]

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::MC_ACRO_R_MAX>)    _param_mc_acro_r_max,
		(ParamFloat<px4::params::MC_ACRO_P_MAX>)    _param_mc_acro_p_max,
		(ParamFloat<px4::params::MC_ACRO_Y_MAX>)    _param_mc_acro_y_max,
		(ParamFloat<px4::params::MC_ACRO_EXPO>)     _param_mc_acro_expo,	///< expo stick curve shape (roll & pitch)
		(ParamFloat<px4::params::MC_ACRO_EXPO_Y>)   _param_mc_acro_expo_y,	///< expo stick curve shape (yaw)
		(ParamFloat<px4::params::MC_ACRO_SUPEXPO>)  _param_mc_acro_supexpo,	///< superexpo stick curve shape (roll & pitch)
		(ParamFloat<px4::params::MC_ACRO_SUPEXPOY>) _param_mc_acro_supexpoy	///< superexpo stick curve shape (yaw)
	)
};
