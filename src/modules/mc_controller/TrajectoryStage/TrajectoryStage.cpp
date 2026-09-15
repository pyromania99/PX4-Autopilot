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

#include "TrajectoryStage.hpp"

#include <float.h>
#include <geo/geo.h>
#include <mathlib/mathlib.h>

using namespace matrix;

/// Tilt fallback when CommandFrontEnd hands over a NaN limit. Matches MPC_TILTMAX_AIR's
/// default of 45 deg.
static constexpr float kDefaultTiltLimit = M_PI_F / 4.f;

/// Smallest cos(tilt) that still counts as "the vertical channel is available". The
/// prototype used 1e-3, which multiplies hover thrust by 1000 and simply pins the output
/// at thrust_max; 0.1 reaches the same saturation with a bounded intermediate.
static constexpr float kMinCosTilt = 0.1f;

TrajectoryStage::TrajectoryStage(ModuleParams *parent) :
	ModuleParams(parent)
{
	TrajectoryStage::updateParams();
	TrajectoryStage::reset();
}

void TrajectoryStage::updateParams()
{
	ModuleParams::updateParams();

	_pos_i = Vector3f(_param_mc_ol_xy_i.get(), _param_mc_ol_xy_i.get(), _param_mc_ol_z_i.get());
	_integral_limit = math::max(_param_mc_ol_i_lim.get(), 0.f);
}

void TrajectoryStage::reset()
{
	_integral.setZero();
	_position_setpoint = Vector3f(NAN, NAN, NAN);
	_velocity_setpoint = Vector3f(NAN, NAN, NAN);
	_acceleration_setpoint = Vector3f(NAN, NAN, NAN);
	_hover_thrust_prev = NAN;
	_stage_valid = false;
}

void TrajectoryStage::resetIntegral()
{
	_integral.setZero();

	// Dropped with the integral: the next hover-thrust change has nothing to correct, and
	// carrying the old value would inject a step into a freshly zeroed integrator.
	_hover_thrust_prev = NAN;
}

void TrajectoryStage::absorbHoverThrustChange(float hover_thrust)
{
	const float h_new = hover_thrust;
	const float h_old = _hover_thrust_prev;

	_hover_thrust_prev = h_new;

	if (!PX4_ISFINITE(h_new) || (h_new < FLT_EPSILON)) {
		return;
	}

	// First sample since a reset: nothing to carry forward.
	if (!PX4_ISFINITE(h_old) || (fabsf(h_new - h_old) < FLT_EPSILON)) {
		return;
	}

	const float a_z = _acceleration_setpoint(2);

	if (!PX4_ISFINITE(a_z)) {
		return;
	}

	_integral(2) += (a_z - CONSTANTS_ONE_G) * (h_old / h_new) + CONSTANTS_ONE_G - a_z;
	_integral(2) = math::constrain(_integral(2), -_integral_limit, _integral_limit);
}

void TrajectoryStage::updateIntegral(const mc_ctrl::ControllerState &state, int axis, float position_error)
{
	// Only integrate on a real position interval. The stage also runs on first entry after
	// a reset, where dt_position describes whatever interval preceded it rather than one
	// this stage observed.
	if (!state.freshness.position_new) {
		return;
	}

	// An estimator jump is not a tracking error. Integrating it would wind the integrator
	// by the size of the discontinuity and then unwind it over the following seconds.
	const bool reset_this_axis = (axis < 2) ? state.resets.xy : state.resets.z;

	if (reset_this_axis) {
		return;
	}

	_integral(axis) += _pos_i(axis) * position_error * state.freshness.dt_position;
	_integral(axis) = math::constrain(_integral(axis), -_integral_limit, _integral_limit);
}

Vector3f TrajectoryStage::computeAccelerationSetpoint(const mc_ctrl::ControllerState &state,
		const mc_ctrl::ControllerCommand &command,
		const Vector3f &pos_p, const Vector3f &pos_d)
{
	// Before anything reads the integral: rescale it for a hover-thrust change, using the
	// PREVIOUS cycle's acceleration setpoint, which is still cached at this point.
	absorbHoverThrustChange(state.hover_thrust);

	// NaN means "not commanded" in a setpoint and "not estimable" in state, so the two have
	// to be tested together. An unset axis contributes no error rather than poisoning the
	// sum. Velocity-only modes deliver a velocity_sp with position_sp NaN, which a
	// position-error-only law cannot serve at all.
	Vector3f acceleration_sp{};

	for (int i = 0; i < 3; i++) {
		const bool have_position = PX4_ISFINITE(command.position_sp(i)) && PX4_ISFINITE(state.position(i));
		const bool have_velocity = PX4_ISFINITE(command.velocity_sp(i)) && PX4_ISFINITE(state.velocity(i));

		const float position_error = have_position ? (command.position_sp(i) - state.position(i)) : 0.f;

		// With no velocity setpoint the D term damps absolute velocity, which is the
		// same thing with v_sp == 0.
		float velocity_error = 0.f;

		if (have_velocity) {
			velocity_error = command.velocity_sp(i) - state.velocity(i);

		} else if (PX4_ISFINITE(state.velocity(i))) {
			velocity_error = -state.velocity(i);
		}

		// Only where a position is actually commanded: on a velocity-only axis there is no
		// hold target, so there is nothing for an integrator to converge onto. This also
		// drains the term the moment the mode stops commanding position.
		if (have_position) {
			updateIntegral(state, i, position_error);

		} else {
			_integral(i) = 0.f;
		}

		acceleration_sp(i) = pos_p(i) * position_error + pos_d(i) * velocity_error + _integral(i);

		if (PX4_ISFINITE(command.acceleration_sp(i))) {
			acceleration_sp(i) += command.acceleration_sp(i);
		}
	}

	// Bound the lateral demand by what the tilt limit can actually deliver, before it
	// reaches the angle construction. g*tan(tilt) is the horizontal acceleration a vehicle
	// at that tilt produces while holding altitude.
	const float tilt_limit = PX4_ISFINITE(command.tilt_limit) ? command.tilt_limit : kDefaultTiltLimit;
	const float max_lateral_acceleration = CONSTANTS_ONE_G * tanf(math::min(tilt_limit, math::radians(80.f)));
	const Vector2f lateral_acceleration(acceleration_sp);
	const float lateral_norm = lateral_acceleration.norm();

	if (lateral_norm > max_lateral_acceleration && lateral_norm > FLT_EPSILON) {
		const Vector2f limited = lateral_acceleration * (max_lateral_acceleration / lateral_norm);
		acceleration_sp(0) = limited(0);
		acceleration_sp(1) = limited(1);
	}

	_position_setpoint = command.position_sp;
	_velocity_setpoint = command.velocity_sp;
	_acceleration_setpoint = acceleration_sp;
	_stage_valid = true;

	return acceleration_sp;
}

float TrajectoryStage::currentHeading(const mc_ctrl::ControllerState &state)
{
	// state.q is documented as always finite while the module runs, but this is also the
	// failsafe controller's path, so it is checked rather than trusted.
	const Quatf q = state.q;
	const bool q_usable = PX4_ISFINITE(q(0)) && PX4_ISFINITE(q(1)) && PX4_ISFINITE(q(2)) && PX4_ISFINITE(q(3))
			      && (q.norm_squared() > FLT_EPSILON);

	if (q_usable) {
		const float heading = Eulerf(q).psi();

		if (PX4_ISFINITE(heading)) {
			return heading;
		}
	}

	return PX4_ISFINITE(state.heading) ? state.heading : 0.f;
}

Vector3f TrajectoryStage::computeThrustSetpoint(const mc_ctrl::ControllerState &state,
		const mc_ctrl::ControllerCommand &command,
		const Vector3f &acceleration_sp) const
{
	// Normalized, not Newtons. Hover thrust is by definition what produces 1 g, so a
	// demand of a m/s^2 costs a/g of it. acceleration_sp(2) is NED down-positive, so
	// climbing (negative) adds thrust and this quantity is negative in normal flight.
	const float thrust_ned_z = acceleration_sp(2) * (state.hover_thrust / CONSTANTS_ONE_G) - state.hover_thrust;

	// Project onto the CURRENT attitude, not the desired one: while the vehicle is still
	// rotating toward the setpoint it is the present tilt that decides how much of the
	// collective ends up vertical. Signed and gated rather than fabsf-and-floored - see
	// the header for why the boost outside the cone is deliberately absent.
	const float cos_tilt = Dcmf(state.q)(2, 2);
	float collective = thrust_ned_z;

	if (cos_tilt > kMinCosTilt) {
		collective = thrust_ned_z / cos_tilt;
	}

	return Vector3f(0.f, 0.f, math::constrain(collective, -command.thrust_max, -command.thrust_min));
}

void TrajectoryStage::fillLocalPositionSetpoint(vehicle_local_position_setpoint_s &sp,
		const Quatf &attitude_sp, const Vector3f &thrust_setpoint) const
{
	sp.x = _position_setpoint(0);
	sp.y = _position_setpoint(1);
	sp.z = _position_setpoint(2);

	sp.vx = _velocity_setpoint(0);
	sp.vy = _velocity_setpoint(1);
	sp.vz = _velocity_setpoint(2);

	_acceleration_setpoint.copyTo(sp.acceleration);

	const Dcmf R_sp(attitude_sp);
	const Vector3f thrust_ned = R_sp * thrust_setpoint;
	thrust_ned.copyTo(sp.thrust);

	// Yaw is not controlled by any law that shares this stage, so reporting a setpoint
	// would misrepresent what the vehicle was asked to do.
	sp.yaw = NAN;
	sp.yawspeed = NAN;
}
