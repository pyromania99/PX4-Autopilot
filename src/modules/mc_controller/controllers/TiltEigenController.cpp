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

#include "TiltEigenController.hpp"

#include <float.h>
#include <geo/geo.h>
#include <mathlib/mathlib.h>

using namespace matrix;

/// Tilt fallback when CommandFrontEnd hands over a NaN limit. Matches MPC_TILTMAX_AIR's
/// default of 45 deg.
static constexpr float kDefaultTiltLimit = M_PI_F / 4.f;

/// Floor on cos(tilt) before it divides the hover collective. Same value and same reasoning
/// as MC_CTRL_ALG=1 and 3: it reaches saturation with a bounded intermediate.
static constexpr float kMinCosTilt = 0.1f;

namespace
{

/**
 * Rotate @p body_unit toward @p world_unit until the angle between them is at most
 * @p max_angle.
 *
 * Ported from ControlMath::limitTilt() rather than called, for the same reason
 * CascadedPdController ports it: ControlMath.cpp is compiled into the PositionControl
 * library, and the controllers here are deliberately free of any link dependency on the
 * stock cascade.
 */
void limitTilt(Vector3f &body_unit, const Vector3f &world_unit, const float max_angle)
{
	const float dot_product_unit = body_unit.dot(world_unit);
	float angle = acosf(math::constrain(dot_product_unit, -1.f, 1.f));
	angle = math::min(angle, max_angle);
	Vector3f rejection = body_unit - (dot_product_unit * world_unit);

	// corner case exactly parallel vectors
	if (rejection.norm_squared() < FLT_EPSILON) {
		rejection(0) = 1.f;
	}

	body_unit = cosf(angle) * world_unit + sinf(angle) * rejection.unit();
}

/**
 * Build an attitude from a desired body-z direction and a heading.
 *
 * Ported from ControlMath::bodyzToAttitude() for the same link-independence reason as
 * limitTilt() above.
 *
 * NOTE this is used for TELEMETRY ONLY here - vehicle_attitude_setpoint needs a full
 * quaternion, and a quaternion needs a heading it does not otherwise have. The control
 * path never reads it back. That separation is the point: MC_CTRL_ALG=3 routes control
 * through a heading-bearing attitude setpoint and relies on the heading cancelling out
 * again, whereas here the heading exists only to make the logged setpoint printable.
 */
Quatf bodyzToAttitude(Vector3f body_z, const float yaw_sp)
{
	// zero vector, no direction, set safe level value
	if (body_z.norm_squared() < FLT_EPSILON) {
		body_z(2) = 1.f;
	}

	body_z.normalize();

	// vector of desired yaw direction in XY plane, rotated by PI/2
	const Vector3f y_C{-sinf(yaw_sp), cosf(yaw_sp), 0.f};

	// desired body_x axis, orthogonal to body_z
	Vector3f body_x = y_C % body_z;

	// keep nose to front while inverted upside down
	if (body_z(2) < 0.f) {
		body_x = -body_x;
	}

	if (fabsf(body_z(2)) < 0.000001f) {
		// desired thrust is in XY plane, set X downside to construct correct matrix,
		// but yaw component will not be used actually
		body_x.zero();
		body_x(2) = 1.f;
	}

	body_x.normalize();

	const Vector3f body_y = body_z % body_x;

	Dcmf R_sp;

	for (int i = 0; i < 3; i++) {
		R_sp(i, 0) = body_x(i);
		R_sp(i, 1) = body_y(i);
		R_sp(i, 2) = body_z(i);
	}

	return Quatf(R_sp);
}

/**
 * Reduced-attitude (tilt-axis) error: the rotation that takes the vehicle's current body-z
 * onto @p body_z_setpoint, expressed as an axis-angle vector in the BODY frame.
 *
 * @param R                 current attitude, NED <- body
 * @param body_z_setpoint   desired body-z direction in NED, unit
 * @return theta * n, where n is the unit rotation axis. n lies in the body x-y plane, so
 *         the third component is structurally zero - there is no yaw component to discard,
 *         which is exactly why this formulation gets "yaw is not controlled" for free.
 *
 * THE FIRST-ORDER IDENTITY that makes MC_EIG_ATT_P's tuning transfer: for a small desired
 * roll phi the desired body-z in the body frame is (0, -sin phi, cos phi), whose cross
 * product with e3 is (sin phi, 0, 0), so the error is (phi, 0, 0) - the Euler roll error,
 * same sign and same magnitude. Pitch is the mirror image. Pinned by
 * TiltEigenControllerTest.TiltErrorMatchesEulerErrorAtSmallAngles.
 */
Vector3f tiltError(const Dcmf &R, const Vector3f &body_z_setpoint)
{
	// Desired thrust axis seen from the body. The vehicle's own body-z is e3 there by
	// definition, which is what collapses the general axis-angle formula to the two lines
	// below and keeps the whole thing free of a chart.
	const Vector3f body_z_sp_body = R.transpose() * body_z_setpoint;

	// e3 x v = (-v_y, v_x, 0). Its norm is sin(theta) for unit vectors, and v_z is
	// cos(theta), so the full signed angle comes from one atan2 and is valid over the
	// entire sphere rather than the half of it an asin would cover.
	const Vector3f axis(-body_z_sp_body(1), body_z_sp_body(0), 0.f);
	const float sin_theta = axis.norm();
	const float cos_theta = body_z_sp_body(2);
	const float theta = atan2f(sin_theta, cos_theta);

	if (sin_theta > FLT_EPSILON) {
		// theta/sin(theta) rescales the cross product from sin(theta) to theta. Near
		// zero the ratio tends to 1 and the expression stays well conditioned; the
		// guard above is for the exactly-degenerate case, not for conditioning.
		return axis * (theta / sin_theta);
	}

	// Aligned (theta ~ 0) or exactly inverted (theta ~ pi). Aligned means no error.
	// Inverted means the axis is genuinely undefined - every axis in the body x-y plane
	// is an equally valid way to fall out of it - so pick one rather than command zero
	// and sit there upside down. Roll is the arbitrary choice; the magnitude is a real
	// pi, so this saturates the rate loop, which is the correct response to being
	// inverted.
	return (cos_theta < 0.f) ? Vector3f(M_PI_F, 0.f, 0.f) : Vector3f();
}

} // namespace

TiltEigenController::TiltEigenController(ModuleParams *parent) :
	MulticopterControllerBase(parent)
{
	TiltEigenController::updateParams();
	TiltEigenController::reset();
}

void TiltEigenController::updateParams()
{
	ModuleParams::updateParams();

	// Cached rather than read per cycle: update() runs at gyro rate and must not touch
	// the parameter system.
	_pos_p = Vector3f(_param_mc_teig_xy_p.get(), _param_mc_teig_xy_p.get(), _param_mc_teig_z_p.get());
	_pos_d = Vector3f(_param_mc_teig_xy_d.get(), _param_mc_teig_xy_d.get(), _param_mc_teig_z_d.get());

	_att_p = _param_mc_teig_att_p.get();

	_wn = _param_mc_teig_wn.get();
	_b = _param_mc_teig_b.get();
	_alpha = _param_mc_teig_alpha.get();
	_beta = _param_mc_teig_beta.get();

	// Floored well above zero: a zero inertia is not a "disable this axis" request, it is
	// a nonsensical rigid body, and it would silently zero the torque on that axis while
	// also zeroing the gyroscopic compensation of the other two.
	_inertia = Vector3f(math::max(_param_mc_teig_ixx.get(), 1e-4f),
			    math::max(_param_mc_teig_iyy.get(), 1e-4f),
			    math::max(_param_mc_teig_izz.get(), 1e-4f));

	_inv_torque_max = 1.f / math::max(_param_mc_teig_trq_max.get(), 1e-3f);
}

void TiltEigenController::reset()
{
	// No integrators, no filters, and - unlike MC_CTRL_ALG=3 - no error history either,
	// because there is no derivative term to seed. What has to be dropped is the cached
	// stage output.
	_body_z_setpoint = Vector3f(0.f, 0.f, 1.f);
	_attitude_setpoint = Quatf();
	_thrust_setpoint.setZero();
	_rate_setpoint.setZero();
	_torque.setZero();

	_position_setpoint = Vector3f(NAN, NAN, NAN);
	_velocity_setpoint = Vector3f(NAN, NAN, NAN);
	_acceleration_setpoint = Vector3f(NAN, NAN, NAN);
	_tilt_error.setZero();
	_position_stage_valid = false;
}

void TiltEigenController::stepTrajectoryToAttitude(const mc_ctrl::ControllerState &state,
		const mc_ctrl::ControllerCommand &command)
{
	// Stage 1: PD on position, damped by the filtered EKF velocity rather than by a
	// numerical derivative of the position error. Identical to MC_CTRL_ALG=3, and for the
	// same two reasons: the setpoint steps flight_mode_manager emits on every mode change
	// would otherwise differentiate into a torque spike, and Altitude and velocity-only
	// modes deliver a velocity_sp with position_sp NaN, which a position-error-only law
	// cannot serve at all.
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

	// Bound the lateral demand by what the tilt limit can actually deliver. g*tan(tilt) is
	// the horizontal acceleration a vehicle at that tilt produces while holding altitude.
	//
	// Note what this does and does NOT do here. limitTilt() below already bounds the
	// DIRECTION exactly, so unlike MC_CTRL_ALG=3 this clamp is not load bearing for the
	// commanded attitude - a saturated demand ends up at the same place either way. It is
	// kept so the acceleration setpoint that reaches the log stays physically meaningful
	// and directly comparable against MC_CTRL_ALG=1 and 3. What is NOT kept is
	// MC_CTRL_ALG=3's SECOND clamp on tilt magnitude, which existed only to undo the
	// overshoot of its small-angle inversion.
	const float tilt_limit = PX4_ISFINITE(command.tilt_limit) ? command.tilt_limit : kDefaultTiltLimit;
	const float max_lateral_acceleration = CONSTANTS_ONE_G * tanf(math::min(tilt_limit, math::radians(80.f)));
	const Vector2f lateral_acceleration(acceleration_sp);
	const float lateral_norm = lateral_acceleration.norm();

	if (lateral_norm > max_lateral_acceleration && lateral_norm > FLT_EPSILON) {
		const Vector2f limited = lateral_acceleration * (max_lateral_acceleration / lateral_norm);
		acceleration_sp(0) = limited(0);
		acceleration_sp(1) = limited(1);
	}

	// Stage 2, DIRECTION. This is the change. Where MC_CTRL_ALG=3 linearizes the hover
	// relation and reads two Euler angles out of it, the desired thrust axis is just a
	// direction: thrust acts along -body_z, so producing a horizontal acceleration a needs
	// body_z tilted the opposite way, and g holds up the vertical component. Exact at any
	// angle, no chart, no small-angle assumption.
	//
	// Sanity: a pure north demand gives body_z leaning south, i.e. nose-down pitch; a pure
	// east demand gives right-wing-down roll. Both asserted by test, because a sign error
	// here is a controller that flies away from its setpoint.
	//
	// Altitude is deliberately NOT folded into this direction - acceleration_sp(2) goes to
	// the collective below, matching MC_CTRL_ALG=1 and 3, so a climb demand does not tilt
	// the vehicle.
	Vector3f body_z(-acceleration_sp(0), -acceleration_sp(1), CONSTANTS_ONE_G);

	// limitTilt() takes UNIT vectors - it feeds the dot product straight into acosf(), so
	// an unnormalized argument silently pins the tilt at exactly the limit every cycle.
	if (body_z.norm_squared() < FLT_EPSILON) {
		body_z = Vector3f(0.f, 0.f, 1.f);

	} else {
		body_z.normalize();
	}

	limitTilt(body_z, Vector3f(0.f, 0.f, 1.f), tilt_limit);
	_body_z_setpoint = body_z;

	// Telemetry only, and the ONLY place a heading enters this controller. The control
	// path below reads _body_z_setpoint, never this quaternion.
	const float heading = PX4_ISFINITE(state.heading) ? state.heading : 0.f;
	_attitude_setpoint = bodyzToAttitude(body_z, heading);

	// Stage 2, MAGNITUDE: normalized, not Newtons, and unchanged from MC_CTRL_ALG=3 - it
	// was already free of any Euler dependence. The hover thrust estimate stands in for
	// the mass/gravity product: hover thrust is by definition what produces 1 g, so a
	// demand of a m/s^2 costs a/g of it.
	//
	// Only the hover term is divided by cos(tilt). acceleration_sp(2) is NED down-positive,
	// so climbing (negative) adds thrust.
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

Vector3f TiltEigenController::tiltToRateSetpoint(const mc_ctrl::ControllerState &state, const float yaw_rate_setpoint)
{
	_tilt_error = tiltError(Dcmf(state.q), _body_z_setpoint);

	// Pure P. There is no derivative of this error anywhere - see the note in the header
	// on why differencing a body-frame vector under yaw rate is a trap, and why MC_TEIG_WN
	// is the damping that replaces it. The consequence worth noting at the call site: this
	// function has no memory and no dt, so it is exact at any update rate and there is
	// nothing here for a mode change to carry across.
	return Vector3f(_att_p * _tilt_error(0), _att_p * _tilt_error(1), yaw_rate_setpoint);
}

Vector3f TiltEigenController::eigenTorque(const Vector3f &rate_setpoint, const Vector3f &angular_velocity) const
{
	// Stage 3, the eigen-dynamics inner loop. UNCHANGED from EigenController - it consumes
	// body-frame rates and inertias and never mentioned an Euler angle, so the change of
	// attitude chart does not reach it. That is the substantive claim this controller
	// makes, and TiltEigenControllerTest pins the cross-coupling and gyroscopic terms
	// against the same expectations MC_CTRL_ALG=3's test uses.
	//
	// The prototype writes this as corrections = B @ (M - A_r) @ [p, q, r, 1]', with
	// B = diag(I), M placing the eigenvalues and A_r the assumed plant. That is feedback
	// linearization: torque = I * (desired omega_dot - assumed omega_dot).
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

	// torque = I * omega_dot_des + omega x (I * omega), with the gyroscopic term derived
	// from the three inertias. The prototype hardcodes this coupling at -Ixx*q*r on roll
	// and +Iyy*p*r on pitch, which for an ordinary planar quadrotor (Izz ~ 2*Ixx) has the
	// OPPOSITE SIGN on roll and therefore adds to the coupling it was meant to cancel. The
	// error is second order in body rate, invisible in gentle hover sweeps and not
	// invisible under the spin that follows a rotor failure. Corrected here as it is in
	// MC_CTRL_ALG=3.
	Vector3f torque;
	torque(0) = _inertia(0) * roll_accel_des + (_inertia(2) - _inertia(1)) * q * r;
	torque(1) = _inertia(1) * pitch_accel_des + (_inertia(0) - _inertia(2)) * p * r;
	torque(2) = _inertia(2) * yaw_accel_des + (_inertia(1) - _inertia(0)) * p * q;

	return torque;
}

bool TiltEigenController::update(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
				 float dt, mc_ctrl::ControllerOutput &output)
{
	// HARD CONTRACT from the interface. There are no integrators and no error history to
	// unwind, but the cached stage output is dropped so a mode change cannot carry a stale
	// thrust direction across.
	if (command.reset_integrals || !state.armed) {
		_rate_setpoint.setZero();
		_position_stage_valid = false;
	}

	Vector3f rate_setpoint{};

	switch (command.level) {
	case mc_ctrl::ControlLevel::Trajectory: {
			// Gated on a fresh local position sample so the position stage keeps
			// mc_pos_control's cadence instead of being pulled up to gyro rate. The
			// reduced-attitude error below still runs every cycle against the live
			// attitude - what the gate holds is the DIRECTION, not the error.
			if (state.freshness.position_new || !_position_stage_valid) {
				stepTrajectoryToAttitude(state, command);
			}

			// Yaw rate setpoint is zero: heading is not controlled at this level.
			rate_setpoint = tiltToRateSetpoint(state, 0.f);
			break;
		}

	case mc_ctrl::ControlLevel::Attitude: {
			// Stabilized/Manual. Take only the DIRECTION out of the pilot's attitude
			// setpoint and discard its heading, which is what makes "no yaw control"
			// hold here too; the yaw stick still reaches the feedforward term, so yaw
			// rate commands do work, unlike MC_CTRL_ALG=1.
			//
			// This is the level where dropping Euler buys the most: MC_CTRL_ALG=3
			// decomposes the pilot's quaternion into phi/theta and is singular at
			// +/-90 deg of pitch, whereas a column of the DCM is just a column.
			//
			// NOT tilt-limited here, deliberately. CommandFrontEnd only populates
			// command.tilt_limit in buildTrajectoryCommand(), so clamping at this
			// level would apply a stale value - or the 45 deg fallback - to a pilot
			// whom MPC_MAN_TILT_MAX may legitimately allow more. mc_manual_mapping
			// has already applied the manual limit upstream.
			_body_z_setpoint = Dcmf(command.attitude_sp).col(2);
			_attitude_setpoint = command.attitude_sp;
			_thrust_setpoint = command.thrust_body_sp;

			rate_setpoint = tiltToRateSetpoint(state, command.yaw_sp_move_rate);
			_position_stage_valid = false;
			break;
		}

	case mc_ctrl::ControlLevel::BodyRate:
		// Acro. No attitude stage to run, so the commanded rates go straight into the
		// eigen inner loop and MC_TEIG_WN / MC_TEIG_B do double duty as rate gains.
		rate_setpoint = command.rate_sp;
		_thrust_setpoint = command.thrust_body_sp;
		_attitude_setpoint = Quatf(NAN, NAN, NAN, NAN);
		_tilt_error.setZero();
		_position_stage_valid = false;
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

	// Unused, and that is the property: with the angle-error derivative gone there is no
	// rate-dependent term anywhere in this law, so the output depends only on the state
	// and the command.
	(void)dt;
	return true;
}

void TiltEigenController::fillLocalPositionSetpoint(vehicle_local_position_setpoint_s &sp) const
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

void TiltEigenController::fillStatus(mc_controller_status_s &status) const
{
	// Same slot layout as MC_CTRL_ALG=3, so a log from one can be compared against the
	// other without remapping. Slots 0 and 1 carry the reduced-attitude error where
	// MC_CTRL_ALG=3 carries roll/pitch angle error; to first order they are the same
	// quantity, which is what makes the comparison meaningful.
	status.debug[0] = _tilt_error(0);
	status.debug[1] = _tilt_error(1);
	status.debug[2] = _rate_setpoint(0);
	status.debug[3] = _rate_setpoint(1);
	status.debug[4] = _rate_setpoint(2);
	// Physical torque, before MC_TEIG_TRQ_MAX: this is what tells you whether the scale is
	// set sensibly, which the normalized value published downstream cannot.
	status.debug[5] = _torque(0);
	status.debug[6] = _torque(1);
	status.debug[7] = _position_stage_valid ? 1.f : 0.f;
}
