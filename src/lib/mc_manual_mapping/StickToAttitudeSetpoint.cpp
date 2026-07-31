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

#include "StickToAttitudeSetpoint.hpp"

#include <AttitudeControlMath.hpp>
#include <mathlib/math/Functions.hpp>
#include <mathlib/math/Limits.hpp>

using namespace matrix;

StickToAttitudeSetpoint::StickToAttitudeSetpoint(ModuleParams *parent) :
	ModuleParams(parent)
{
	// Rate of change 5% per second -> 1.6 seconds to ramp to default 8% MPC_MANTHR_MIN
	_manual_throttle_minimum.setSlewRate(0.05f);
	// Rate of change 50% per second -> 2 seconds to ramp to 100%
	_manual_throttle_maximum.setSlewRate(0.5f);
	// Rate of change 5% per second -> 6 seconds to ramp 30% if hover thrust parameter is off
	_hover_thrust_slew_rate.setSlewRate(0.05f);

	StickToAttitudeSetpoint::updateParams();
}

void StickToAttitudeSetpoint::updateParams()
{
	ModuleParams::updateParams();

	// Update from hover thrust parameter if there's no valid estimate in use
	if (!PX4_ISFINITE(_hover_thrust_estimate)) {
		_hover_thrust_slew_rate.setForcedValue(_param_mpc_thr_hover.get());
	}

	_man_tilt_max = math::radians(_param_mpc_man_tilt_max.get());
}

void StickToAttitudeSetpoint::setHoverThrustEstimate(float hover_thrust)
{
	if (PX4_ISFINITE(hover_thrust)) {
		_hover_thrust_estimate = math::constrain(hover_thrust, .05f, .9f);

	} else {
		// Possibly bad estimate before it got invalid, slew back to parameter
		_hover_thrust_estimate = _param_mpc_thr_hover.get();
	}
}

void StickToAttitudeSetpoint::updateSlewRates(bool landed, bool spooled_up, float dt)
{
	if (landed) {
		_manual_throttle_minimum.update(0.f, dt);

	} else {
		_manual_throttle_minimum.update(_param_mpc_manthr_min.get(), dt);
	}

	if (spooled_up) {
		_manual_throttle_maximum.update(1.f, dt);

	} else {
		_manual_throttle_maximum.setForcedValue(0.f);
	}

	if (PX4_ISFINITE(_hover_thrust_estimate)) {
		_hover_thrust_slew_rate.update(_hover_thrust_estimate, dt);
	}
}

void StickToAttitudeSetpoint::ekfResetHandler(float delta_psi)
{
	// Only offset the yaw setpoint when the heading is locked
	if (PX4_ISFINITE(_yaw_setpoint_stabilized)) {
		_yaw_setpoint_stabilized = wrap_pi(_yaw_setpoint_stabilized + delta_psi);
	}

	_stick_yaw.ekfResetHandler(delta_psi);
}

void StickToAttitudeSetpoint::reset(const Quatf &q, float unaided_heading)
{
	_man_roll_input_filter.reset(0.f);
	_man_pitch_input_filter.reset(0.f);
	_yaw_setpoint_stabilized = NAN;
	_stick_yaw.reset(Eulerf(q).psi(), unaided_heading);
}

float StickToAttitudeSetpoint::throttleCurve(float throttle_stick_input) const
{
	float thrust = 0.f;

	// throttle_stick_input is in range [-1, 1]
	switch (_param_mpc_thr_curve.get()) {
	case 1: // no rescaling
		thrust = math::interpolate(throttle_stick_input, -1.f, 1.f,
					   _manual_throttle_minimum.getState(), _param_mpc_thr_max.get());
		break;

	case 2: // rescale to hover thrust param at 0 stick input
		thrust = math::interpolateNXY(throttle_stick_input,
		{-1.f, 0.f, 1.f},
		{_manual_throttle_minimum.getState(), _param_mpc_thr_hover.get(), _param_mpc_thr_max.get()});
		break;

	default: // 0 or other: rescale to HTE value
		thrust = math::interpolateNXY(throttle_stick_input,
		{-1.f, 0.f, 1.f},
		{_manual_throttle_minimum.getState(), _hover_thrust_slew_rate.getState(), _param_mpc_thr_max.get()});
		break;
	}

	return math::min(thrust, _manual_throttle_maximum.getState());
}

void StickToAttitudeSetpoint::update(const manual_control_setpoint_s &manual_control_setpoint, const Quatf &q,
				     float unaided_heading, float dt, vehicle_attitude_setpoint_s &attitude_setpoint)
{
	// Avoid accumulating absolute yaw error with arming stick gesture
	const bool arming_gesture = (manual_control_setpoint.throttle < -.9f) && (_param_mc_airmode.get() != 2);

	if (arming_gesture) {
		_yaw_setpoint_stabilized = NAN;
	}

	const float yaw = Eulerf(q).psi();
	const float yaw_stick_input = math::expo_deadzone(manual_control_setpoint.yaw, .6f, _param_man_deadzone.get());
	_stick_yaw.generateYawSetpoint(attitude_setpoint.yaw_sp_move_rate, _yaw_setpoint_stabilized, yaw_stick_input, yaw, dt,
				       unaided_heading);

	/*
	 * Input mapping for roll & pitch setpoints
	 * ----------------------------------------
	 * We control the following 2 angles:
	 * - tilt angle, given by sqrt(roll*roll + pitch*pitch)
	 * - the direction of the maximum tilt in the XY-plane, which also defines the direction of the motion
	 *
	 * This allows a simple limitation of the tilt angle, the vehicle flies towards the direction that the stick
	 * points to, and changes of the stick input are linear.
	 */
	_man_roll_input_filter.setParameters(dt, _param_mc_man_tilt_tau.get());
	_man_pitch_input_filter.setParameters(dt, _param_mc_man_tilt_tau.get());

	// we want to fly towards the direction of (roll, pitch)
	Vector2f v = Vector2f(_man_roll_input_filter.update(manual_control_setpoint.roll * _man_tilt_max),
			      -_man_pitch_input_filter.update(manual_control_setpoint.pitch * _man_tilt_max));
	float v_norm = v.norm(); // the norm of v defines the tilt angle

	if (v_norm > _man_tilt_max) { // limit to the configured maximum tilt angle
		v *= _man_tilt_max / v_norm;
	}

	Quatf q_sp_rp = AxisAnglef(v(0), v(1), 0.f);
	// Make sure there's a valid attitude quaternion with no yaw error when yaw is unlocked (NAN)
	const float yaw_setpoint = PX4_ISFINITE(_yaw_setpoint_stabilized) ? _yaw_setpoint_stabilized : yaw;
	// The axis angle can change the yaw as well (noticeable at higher tilt angles).
	// This is the formula by how much the yaw changes:
	//   let a := tilt angle, b := atan(y/x) (direction of maximum tilt)
	//   yaw = atan(-2 * sin(b) * cos(b) * sin^2(a/2) / (1 - 2 * cos^2(b) * sin^2(a/2))).
	const Quatf q_sp_yaw(cosf(yaw_setpoint / 2.f), 0.f, 0.f, sinf(yaw_setpoint / 2.f));

	if (_vtol) {
		// Modify the setpoints for roll and pitch such that they reflect the user's intention even
		// if a large yaw error(yaw_sp - yaw) is present. In the presence of a yaw error constructing
		// an attitude setpoint from the yaw setpoint will lead to unexpected attitude behaviour from
		// the user's view as the tilt will not be aligned with the heading of the vehicle.

		AttitudeControlMath::correctTiltSetpointForYawError(q_sp_rp, q, q_sp_yaw);
	}

	// Align the desired tilt with the yaw setpoint
	Quatf q_sp = q_sp_yaw * q_sp_rp;

	q_sp.copyTo(attitude_setpoint.q_d);

	attitude_setpoint.thrust_body[2] = -throttleCurve(manual_control_setpoint.throttle);
}
