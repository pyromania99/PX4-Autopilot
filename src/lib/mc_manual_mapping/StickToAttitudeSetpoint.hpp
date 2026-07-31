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
 * @file StickToAttitudeSetpoint.hpp
 * @brief Map manual stick input to an attitude setpoint (Manual/Stabilized mode).
 *
 * Extracted verbatim from MulticopterAttitudeControl::generate_attitude_setpoint()
 * and ::throttle_curve() so that both the stock mc_att_control module and the
 * pluggable controller framework share one implementation of the stick mapping.
 * Behaviour must remain bit-identical to the pre-extraction module; see
 * StickToAttitudeSetpointTest.cpp which differentially compares this class
 * against a verbatim copy of the original code.
 */

#pragma once

#include <lib/mathlib/math/filter/AlphaFilter.hpp>
#include <lib/slew_rate/SlewRate.hpp>
#include <lib/stick_yaw/StickYaw.hpp>
#include <matrix/matrix/math.hpp>
#include <px4_platform_common/module_params.h>

#include <uORB/topics/manual_control_setpoint.h>
#include <uORB/topics/vehicle_attitude_setpoint.h>

class StickToAttitudeSetpoint : public ModuleParams
{
public:
	explicit StickToAttitudeSetpoint(ModuleParams *parent);
	~StickToAttitudeSetpoint() override = default;

	/**
	 * Enable the VTOL tilt correction (AttitudeControlMath::correctTiltSetpointForYawError).
	 */
	void setVtol(bool vtol) { _vtol = vtol; }

	/**
	 * Feed a new hover thrust estimate. Call this ONLY when the hover_thrust_estimate
	 * subscription actually updated, to preserve the original semantics where the
	 * estimate stays NAN until the first message arrives.
	 * @param hover_thrust valid estimate, or NAN when the estimate is flagged invalid
	 */
	void setHoverThrustEstimate(float hover_thrust);

	/**
	 * Advance the throttle min/max and hover thrust slew rates. Call once per cycle.
	 */
	void updateSlewRates(bool landed, bool spooled_up, float dt);

	/**
	 * Offset the held yaw setpoint by an EKF heading reset.
	 */
	void ekfResetHandler(float delta_psi);

	/**
	 * Drop the held yaw setpoint and the tilt input filters. Call when leaving
	 * manual-attitude control so re-entry does not resume a stale setpoint.
	 */
	void reset(const matrix::Quatf &q, float unaided_heading);

	/**
	 * Generate an attitude setpoint from stick input.
	 * Fills @p attitude_setpoint; it does NOT publish and does NOT set the timestamp.
	 */
	void update(const manual_control_setpoint_s &manual_control_setpoint, const matrix::Quatf &q,
		    float unaided_heading, float dt, vehicle_attitude_setpoint_s &attitude_setpoint);

protected:
	void updateParams() override;

private:
	float throttleCurve(float throttle_stick_input) const;

	StickYaw _stick_yaw{this};

	float _hover_thrust_estimate{NAN};
	SlewRate<float> _hover_thrust_slew_rate{.5f};

	float _yaw_setpoint_stabilized{0.f};
	float _man_tilt_max{0.f};			///< maximum tilt allowed for manual flight [rad]

	SlewRate<float> _manual_throttle_minimum{0.f};	///< 0 when landed and ramped to MPC_MANTHR_MIN in air
	SlewRate<float> _manual_throttle_maximum{0.f};	///< 0 when disarmed ramped to 1 when spooled up
	AlphaFilter<float> _man_roll_input_filter;
	AlphaFilter<float> _man_pitch_input_filter;

	bool _vtol{false};

	DEFINE_PARAMETERS(
		(ParamInt<px4::params::MC_AIRMODE>)         _param_mc_airmode,
		(ParamFloat<px4::params::MC_MAN_TILT_TAU>)  _param_mc_man_tilt_tau,

		(ParamFloat<px4::params::MAN_DEADZONE>)     _param_man_deadzone,
		(ParamFloat<px4::params::MPC_MAN_TILT_MAX>) _param_mpc_man_tilt_max,
		(ParamFloat<px4::params::MPC_MANTHR_MIN>)   _param_mpc_manthr_min,
		(ParamFloat<px4::params::MPC_THR_MAX>)      _param_mpc_thr_max,
		(ParamFloat<px4::params::MPC_THR_HOVER>)    _param_mpc_thr_hover,
		(ParamInt<px4::params::MPC_THR_CURVE>)      _param_mpc_thr_curve
	)
};
