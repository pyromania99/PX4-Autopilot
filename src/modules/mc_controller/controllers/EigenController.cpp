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

#include "EigenController.hpp"

#include <float.h>
#include <geo/geo.h>
#include <mathlib/mathlib.h>

using namespace matrix;

/// Tilt fallback when CommandFrontEnd hands over a NaN limit. Matches MPC_TILTMAX_AIR's
/// default of 45 deg.
static constexpr float kDefaultTiltLimit = M_PI_F / 4.f;

/// Floor on cos(tilt) before it divides the hover collective. The prototype used 1e-3,
/// which multiplies hover thrust by 1000 and simply pins the output at thrust_max; 0.1
/// reaches the same saturation with a bounded intermediate, and matches
/// CascadedPdController.
static constexpr float kMinCosTilt = 0.1f;

EigenController::EigenController(ModuleParams *parent) :
	MulticopterControllerBase(parent)
{
	EigenController::updateParams();
	EigenController::reset();
}

void EigenController::updateParams()
{
	ModuleParams::updateParams();

	// Cached rather than read per cycle: update() runs at gyro rate and must not touch
	// the parameter system.
	_pos_p = Vector3f(_param_mc_eig_xy_p.get(), _param_mc_eig_xy_p.get(), _param_mc_eig_z_p.get());
	_pos_d = Vector3f(_param_mc_eig_xy_d.get(), _param_mc_eig_xy_d.get(), _param_mc_eig_z_d.get());

	_att_p = _param_mc_eig_att_p.get();
	_att_d = _param_mc_eig_att_d.get();

	_wn = _param_mc_eig_wn.get();
	_b = _param_mc_eig_b.get();
	_alpha = _param_mc_eig_alpha.get();
	_beta = _param_mc_eig_beta.get();

	// Floored well above zero: a zero inertia is not a "disable this axis" request, it is
	// a nonsensical rigid body, and it would silently zero the torque on that axis while
	// also zeroing the gyroscopic compensation of the other two.
	_inertia = Vector3f(math::max(_param_mc_eig_ixx.get(), 1e-4f),
			    math::max(_param_mc_eig_iyy.get(), 1e-4f),
			    math::max(_param_mc_eig_izz.get(), 1e-4f));

	_inv_torque_max = 1.f / math::max(_param_mc_eig_trq_max.get(), 1e-3f);
}

void EigenController::reset()
{
	// No integrators and no filters. What has to be dropped is the cached stage output
	// and, crucially, the angle-error history: a stale error from the previous mode
	// differentiates into exactly the spike _first_attitude_update exists to prevent.
	_roll_setpoint = 0.f;
	_pitch_setpoint = 0.f;
	_attitude_setpoint = Quatf();
	_thrust_setpoint.setZero();
	_rate_setpoint.setZero();
	_torque.setZero();

	_roll_error_prev = 0.f;
	_pitch_error_prev = 0.f;
	_roll_error_rate = 0.f;
	_pitch_error_rate = 0.f;
	_first_attitude_update = true;

	_position_setpoint = Vector3f(NAN, NAN, NAN);
	_velocity_setpoint = Vector3f(NAN, NAN, NAN);
	_acceleration_setpoint = Vector3f(NAN, NAN, NAN);
	_attitude_error.setZero();
	_position_stage_valid = false;
}

void EigenController::stepTrajectoryToAttitude(const mc_ctrl::ControllerState &state,
		const mc_ctrl::ControllerCommand &command)
{
	// Stage 1: PD on position, damped by the filtered EKF velocity rather than by the
	// prototype's numerical derivative of the position error. Two reasons, the same ones
	// CascadedPdController gives: the setpoint steps flight_mode_manager emits on every
	// mode change would otherwise differentiate into a torque spike, and Altitude and
	// velocity-only modes deliver a velocity_sp with position_sp NaN, which a
	// position-error-only law cannot serve at all.
	//
	// An unset (NaN) axis contributes no error rather than poisoning the sum. There is no
	// integrator anywhere in this controller: steady wind leaves a steady offset.
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

		acceleration_sp(i) = _pos_p(i) * position_error + _pos_d(i) * velocity_error;

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

	// Stage 2, direction. The prototype inverts the small-angle hover relation in ENU/FLU;
	// re-derived here for NED/FRD, where the third column of R(phi, theta, psi) is
	// approximately [theta*cos(psi) + phi*sin(psi), theta*sin(psi) - phi*cos(psi), 1] and
	// horizontal acceleration is -g times its first two entries:
	//
	//   theta_des = -(a_n*cos(psi) + a_e*sin(psi)) / g
	//   phi_des   =  (a_e*cos(psi) - a_n*sin(psi)) / g
	//
	// Sanity: a pure north demand at psi = 0 gives nose-down pitch and zero roll; a pure
	// east demand gives right-wing-down roll and zero pitch. Both are asserted by test,
	// because a sign error here is a controller that flies away from its setpoint.
	//
	// THE HEADING DECISION, same as MC_CTRL_ALG=1: state.heading, not command.yaw_sp. The
	// tilt is rebuilt around wherever the vehicle points right now, so a heading error can
	// never appear and the yaw axis never sees a proportional term.
	const float heading = PX4_ISFINITE(state.heading) ? state.heading : 0.f;
	const float sin_heading = sinf(heading);
	const float cos_heading = cosf(heading);

	float pitch_sp = -(acceleration_sp(0) * cos_heading + acceleration_sp(1) * sin_heading) / CONSTANTS_ONE_G;
	float roll_sp = (acceleration_sp(1) * cos_heading - acceleration_sp(0) * sin_heading) / CONSTANTS_ONE_G;

	// The linear inversion overshoots the true angle: clamping lateral acceleration to
	// g*tan(tilt) above leaves it free to ask for tan(tilt) RADIANS, which at a 45 deg
	// limit is 57 deg. Clamp the tilt magnitude too, so the framework's limit is what
	// actually reaches the vehicle.
	const float tilt_magnitude = sqrtf(roll_sp * roll_sp + pitch_sp * pitch_sp);

	if (tilt_magnitude > tilt_limit && tilt_magnitude > FLT_EPSILON) {
		const float scale = tilt_limit / tilt_magnitude;
		roll_sp *= scale;
		pitch_sp *= scale;
	}

	_roll_setpoint = roll_sp;
	_pitch_setpoint = pitch_sp;
	_attitude_setpoint = Quatf(Eulerf(roll_sp, pitch_sp, heading));

	// Stage 2, magnitude: normalized, not Newtons. The hover thrust estimate stands in for
	// the mass/gravity product the prototype knew exactly - hover thrust is by definition
	// what produces 1 g, so a demand of a m/s^2 costs a/g of it.
	//
	// Only the hover term is divided by cos(tilt), matching the prototype's
	// base_thrust = m*g/(4*tilt_den) with its altitude correction left undivided.
	// acceleration_sp(2) is NED down-positive, so climbing (negative) adds thrust.
	const float cos_tilt = math::max(fabsf(Dcmf(state.q)(2, 2)), kMinCosTilt);
	const float collective = state.hover_thrust / cos_tilt
				 - acceleration_sp(2) * state.hover_thrust / CONSTANTS_ONE_G;

	_thrust_setpoint = Vector3f(0.f, 0.f,
				    -math::constrain(collective, command.thrust_min, command.thrust_max));

	_position_setpoint = command.position_sp;
	_velocity_setpoint = command.velocity_sp;
	_acceleration_setpoint = acceleration_sp;
	_position_stage_valid = true;
}

Vector3f EigenController::angleToRateSetpoint(const mc_ctrl::ControllerState &state, const float yaw_rate_setpoint)
{
	const Eulerf euler(state.q);
	const float roll_error = wrap_pi(_roll_setpoint - euler.phi());
	const float pitch_error = wrap_pi(_pitch_setpoint - euler.theta());

	// The derivative is refreshed only on a new attitude sample and held in between. At
	// gyro rate the attitude has usually not changed since the last call, so
	// differentiating every cycle would divide estimator quantization by a very small dt
	// and inject that straight into the rate setpoint.
	if (_first_attitude_update) {
		// Seed from the current error so the first step produces a zero derivative
		// rather than (error - 0) / dt. The prototype's _first_update guard.
		_roll_error_prev = roll_error;
		_pitch_error_prev = pitch_error;
		_roll_error_rate = 0.f;
		_pitch_error_rate = 0.f;
		_first_attitude_update = false;

	} else if (state.freshness.attitude_new && state.freshness.dt_attitude > FLT_EPSILON) {
		const float inv_dt = 1.f / state.freshness.dt_attitude;
		_roll_error_rate = (roll_error - _roll_error_prev) * inv_dt;
		_pitch_error_rate = (pitch_error - _pitch_error_prev) * inv_dt;
		_roll_error_prev = roll_error;
		_pitch_error_prev = pitch_error;
	}

	_attitude_error = Vector2f(roll_error, pitch_error);

	return Vector3f(_att_p * roll_error + _att_d * _roll_error_rate,
			_att_p * pitch_error + _att_d * _pitch_error_rate,
			yaw_rate_setpoint);
}

Vector3f EigenController::eigenTorque(const Vector3f &rate_setpoint, const Vector3f &angular_velocity) const
{
	// Stage 3, the eigen-dynamics inner loop. The prototype writes this as
	//
	//   corrections = B @ (M - A_r) @ [p, q, r, 1]'
	//
	// with B = diag(I), M = [M_dyn | M_des], M_dyn placing the eigenvalues and A_r the
	// assumed plant. That is feedback linearization: torque = I * (desired omega_dot -
	// assumed omega_dot). Expanded, the 3x4 product is the handful of terms below, which
	// is what runs here - a matrix multiply at gyro rate to reach six useful numbers is
	// not a trade worth making.
	const float p = angular_velocity(0);
	const float q = angular_velocity(1);
	const float r = angular_velocity(2);

	const float roll_rate_error = rate_setpoint(0) - p;
	const float pitch_rate_error = rate_setpoint(1) - q;

	// Desired angular acceleration. The cross terms (+b, -b) are the imaginary part of the
	// eigenvalue pair -wn +/- j*b: this is what couples the two axes into one rotating
	// mode. The +alpha terms cancel the rate damping the airframe is assumed to have.
	const float roll_accel_des = _wn * roll_rate_error + _b * pitch_rate_error + _alpha * p;
	const float pitch_accel_des = _wn * pitch_rate_error - _b * roll_rate_error + _alpha * q;

	// Yaw carries no feedback term. In the prototype's algebra M_dyn(2,2) = -beta and
	// A_r(2,2) = -beta cancel exactly, leaving pure feedforward - cancelling the assumed
	// yaw drag and imposing the same eigenvalue is a no-op on the measured rate. So a zero
	// yaw rate setpoint commands zero yaw torque, which is the free-running heading the
	// design wants, and it is a property of the law rather than a special case bolted on.
	const float yaw_accel_des = _beta * rate_setpoint(2);

	// torque = I * omega_dot_des + omega x (I * omega). The gyroscopic term is derived
	// from the three inertias rather than hardcoded, and this is the one place the port
	// deliberately corrects the prototype instead of reproducing it.
	//
	// The prototype's A_r fixes the coupling at -Ixx*q*r on roll and +Iyy*p*r on pitch,
	// with nothing at all on yaw. Matching omega x (I*omega) would need Ixx == Iyy AND
	// Izz == 0 simultaneously, which no rigid body satisfies. For an ordinary quadrotor,
	// where the rotors lie in a plane and Izz is about 2*Ixx, the true roll term is
	// +Ixx*q*r - the prototype's has the OPPOSITE SIGN, so it adds to the coupling it was
	// meant to cancel. The error is second order in body rate, which is why it stayed
	// invisible in gentle hover sweeps and would not have stayed invisible under the spin
	// that follows a rotor failure.
	//
	// EigenControllerTest pins this against a direct omega x (I*omega) evaluation.
	Vector3f torque;
	torque(0) = _inertia(0) * roll_accel_des + (_inertia(2) - _inertia(1)) * q * r;
	torque(1) = _inertia(1) * pitch_accel_des + (_inertia(0) - _inertia(2)) * p * r;
	torque(2) = _inertia(2) * yaw_accel_des + (_inertia(1) - _inertia(0)) * p * q;

	return torque;
}

bool EigenController::update(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
			     float dt, mc_ctrl::ControllerOutput &output)
{
	// HARD CONTRACT from the interface. There is nothing to unwind in a law with no
	// integrators, but the cached stage outputs and the angle-error history are dropped so
	// a mode change cannot carry either across.
	if (command.reset_integrals || !state.armed) {
		_rate_setpoint.setZero();
		_position_stage_valid = false;
		_first_attitude_update = true;
	}

	Vector3f rate_setpoint{};

	switch (command.level) {
	case mc_ctrl::ControlLevel::Trajectory: {
			// Gated on a fresh local position sample so the position stage keeps
			// mc_pos_control's cadence instead of being pulled up to gyro rate.
			if (state.freshness.position_new || !_position_stage_valid) {
				stepTrajectoryToAttitude(state, command);
			}

			// Yaw rate setpoint is zero: heading is not controlled at this level. See
			// the file comment.
			rate_setpoint = angleToRateSetpoint(state, 0.f);
			break;
		}

	case mc_ctrl::ControlLevel::Attitude: {
			// Stabilized/Manual. Take the pilot's tilt through the same Euler angle PD
			// the trajectory stage feeds, and let the yaw stick reach the feedforward
			// term - unlike MC_CTRL_ALG=1, yaw rate commands do work here, because the
			// law has somewhere to put them.
			const Eulerf euler_sp(command.attitude_sp);
			_roll_setpoint = euler_sp.phi();
			_pitch_setpoint = euler_sp.theta();
			_attitude_setpoint = command.attitude_sp;
			_thrust_setpoint = command.thrust_body_sp;

			rate_setpoint = angleToRateSetpoint(state, command.yaw_sp_move_rate);
			_position_stage_valid = false;
			break;
		}

	case mc_ctrl::ControlLevel::BodyRate:
		// Acro. No angle stage to run, so the commanded rates go straight into the
		// eigen inner loop and MC_EIG_WN / MC_EIG_B do double duty as rate gains.
		rate_setpoint = command.rate_sp;
		_thrust_setpoint = command.thrust_body_sp;
		_attitude_setpoint = Quatf(NAN, NAN, NAN, NAN);
		_attitude_error.setZero();
		_position_stage_valid = false;
		// The angle PD is not running, so its error history is stale the moment we
		// return to a level that uses it.
		_first_attitude_update = true;
		break;

	case mc_ctrl::ControlLevel::None:
	default:
		return false;
	}

	_rate_setpoint = rate_setpoint;
	_torque = eigenTorque(rate_setpoint, state.angular_velocity);

	// N m -> normalized. control_allocator normalizes its own mix columns, so what it
	// wants here is dimensionless and physical torque would be silently misinterpreted.
	Vector3f torque = _torque * _inv_torque_max;

	for (int i = 0; i < 3; i++) {
		torque(i) = PX4_ISFINITE(torque(i)) ? math::constrain(torque(i), -1.f, 1.f) : 0.f;
	}

	for (int i = 0; i < 3; i++) {
		if (!PX4_ISFINITE(_thrust_setpoint(i))) {
			_thrust_setpoint(i) = 0.f;
		}
	}

	output.torque = torque;
	output.thrust = _thrust_setpoint;
	output.rate_setpoint = _rate_setpoint;
	output.attitude_setpoint = _attitude_setpoint;
	output.valid = true;

	(void)dt;	// the only rate-dependent term uses dt_attitude, not the gyro dt
	return true;
}

void EigenController::fillLocalPositionSetpoint(vehicle_local_position_setpoint_s &sp) const
{
	// Telemetry only. Stock fills this from PositionControl's internal setpoints; here it
	// comes from what the PD stage actually used, so a log still shows what was asked for
	// versus what was flown.
	sp.x = _position_setpoint(0);
	sp.y = _position_setpoint(1);
	sp.z = _position_setpoint(2);
	sp.vx = _velocity_setpoint(0);
	sp.vy = _velocity_setpoint(1);
	sp.vz = _velocity_setpoint(2);
	_acceleration_setpoint.copyTo(sp.acceleration);

	// NED thrust, matching PositionControl::getLocalPositionSetpoint().
	const Dcmf R_sp(_attitude_setpoint);
	const Vector3f thrust_ned = R_sp * _thrust_setpoint;
	thrust_ned.copyTo(sp.thrust);

	// Heading is not controlled at Trajectory level, so there is no yaw setpoint to report.
	sp.yaw = NAN;
	sp.yawspeed = NAN;
}

void EigenController::fillStatus(mc_controller_status_s &status) const
{
	status.debug[0] = _attitude_error(0);
	status.debug[1] = _attitude_error(1);
	status.debug[2] = _rate_setpoint(0);
	status.debug[3] = _rate_setpoint(1);
	status.debug[4] = _rate_setpoint(2);
	// Physical torque, before MC_EIG_TRQ_MAX: this is what tells you whether the scale is
	// set sensibly, which the normalized value published downstream cannot.
	status.debug[5] = _torque(0);
	status.debug[6] = _torque(1);
	status.debug[7] = _position_stage_valid ? 1.f : 0.f;
}
