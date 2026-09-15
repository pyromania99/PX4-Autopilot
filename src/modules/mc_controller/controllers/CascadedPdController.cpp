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
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

#include "CascadedPdController.hpp"

#include <float.h>
#include <geo/geo.h>
#include <mathlib/mathlib.h>

using namespace matrix;

/// Tilt fallback when CommandFrontEnd hands over a NaN limit. Matches MPC_TILTMAX_AIR's
/// default of 45 deg.
static constexpr float kDefaultTiltLimit = M_PI_F / 4.f;

namespace
{

/**
 * Rotate @p body_unit toward @p world_unit until the angle between them is at most
 * @p max_angle.
 *
 * Ported from ControlMath::limitTilt() rather than called: ControlMath.cpp is compiled
 * into the PositionControl library, and this controller is deliberately free of any link
 * dependency on the stock cascade.
 */
void limitTilt(Vector3f &body_unit, const Vector3f &world_unit, const float max_angle)
{
	const float dot_product_unit = body_unit.dot(world_unit);
	float angle = acosf(dot_product_unit);
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
 * This is the Z_b_des / X_c_des / Y_b_des cross-product construction the Isaac Sim
 * prototype writes out by hand, ported from ControlMath::bodyzToAttitude() for the same
 * link-independence reason as limitTilt() above. The two extra branches over the
 * prototype are the inverted case and the thrust-in-the-XY-plane case, which the
 * prototype's `Y_tmp_norm < 1e-6` guard handles less completely.
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

} // namespace

CascadedPdController::CascadedPdController(ModuleParams *parent) :
	MulticopterControllerBase(parent),
	_trajectory_stage(this)
{
	CascadedPdController::updateParams();
	CascadedPdController::reset();
}

void CascadedPdController::updateParams()
{
	ModuleParams::updateParams();

	// Cached rather than read per cycle: update() runs at gyro rate and must not touch
	// the parameter system.
	_pos_p = Vector3f(_param_mc_pd_xy_p.get(), _param_mc_pd_xy_p.get(), _param_mc_pd_z_p.get());
	_pos_d = Vector3f(_param_mc_pd_xy_d.get(), _param_mc_pd_xy_d.get(), _param_mc_pd_z_d.get());
	_att_p = _param_mc_pd_att_p.get();

	// Roll/pitch damping is floored above zero. The inner loop factors the law as
	// Kd*(w_sp - w) with w_sp = -(Kp/Kd)*e_R, which is exact for any Kd > 0 but
	// collapses to zero torque at Kd == 0 - silently disabling roll and pitch rather
	// than leaving the undamped -Kp*e_R the parameter appears to ask for. Kd == 0 is an
	// unflyable setting either way, so clamp it instead of carrying a second code path.
	// The yaw axis is NOT floored: MC_PD_YAWR_D == 0 is the default and means exactly
	// what it says, no yaw actuation at all.
	const float att_d = math::max(_param_mc_pd_att_d.get(), 0.01f);
	_att_d = Vector3f(att_d, att_d, _param_mc_pd_yawr_d.get());
}

void CascadedPdController::reset()
{
	// The inner loop has no integrators and no filters, so a reset here is just dropping
	// the cached stage outputs. The shared stage's validity flag is what stops a stale
	// attitude setpoint from the previous mode being commanded before the position stage
	// has run once; its full reset() also drops the position integrator.
	_attitude_setpoint = Quatf();
	_body_z_setpoint = Vector3f(0.f, 0.f, 1.f);
	_thrust_setpoint.setZero();
	_rate_setpoint.setZero();
	_autotune_rate_sp.setZero();
	_attitude_error.setZero();
	_trajectory_stage.reset();
}

void CascadedPdController::bodyzToAttitudeSetpoint(Vector3f body_z, const mc_ctrl::ControllerState &state,
		const mc_ctrl::ControllerCommand &command)
{
	const float tilt_limit = PX4_ISFINITE(command.tilt_limit) ? command.tilt_limit : kDefaultTiltLimit;

	// limitTilt() takes UNIT vectors - it feeds the dot product straight into acosf(),
	// so an unnormalized argument silently yields NaN and pins the tilt at exactly the
	// limit on every cycle. Normalize first, as PositionControl::_accelerationControl()
	// does at its own call site.
	if (body_z.norm_squared() < FLT_EPSILON) {
		body_z = Vector3f(0.f, 0.f, 1.f);

	} else {
		body_z.normalize();
	}

	// Belt and braces on top of the lateral-acceleration clamp in
	// stepTrajectoryToAttitude(): that one bounds what the PD asks for, this one bounds
	// what actually gets commanded, including at the Attitude level where the PD had no
	// say at all.
	limitTilt(body_z, Vector3f(0.f, 0.f, 1.f), tilt_limit);

	// THE heading decision: the vehicle's own heading, not command.yaw_sp. R_des is rebuilt
	// around wherever the vehicle is pointing right now, so a heading error can never appear
	// and the yaw axis never sees a proportional term. See the file comment.
	//
	// Taken from state.q via the shared helper, not from state.heading: the two are the same
	// angle but state.heading is only as fresh as the EKF2 fusion step, and this call site is
	// what makes its age a control error rather than a telemetry detail. See
	// TrajectoryStage::currentHeading().
	_attitude_setpoint = bodyzToAttitude(body_z, TrajectoryStage::currentHeading(state));
}

void CascadedPdController::stepTrajectoryToBodyZ(const mc_ctrl::ControllerState &state,
		const mc_ctrl::ControllerCommand &command)
{
	// Stage 1 and the lateral clamp live in the shared trajectory stage: they were
	// identical across every law in this module, and that is now where the position
	// integrator lives too. The gains stay here - MC_PD_* is this law's own tuning, and its
	// vertical defaults differ from the eigen law's.
	//
	// bodyzToAttitudeSetpoint() re-derives the tilt limit for its own exact directional
	// clamp, so nothing here needs it.
	//
	// This function deliberately stops short of R_des. Everything it computes depends on the
	// position estimate and is therefore only worth recomputing at position rate; the heading
	// R_des is anchored on is not, and update() re-applies that every cycle.
	const Vector3f acceleration_sp = _trajectory_stage.computeAccelerationSetpoint(state, command, _pos_p, _pos_d);

	// Stage 2, direction: the specific force the vehicle must produce is
	// a_des - g_ned, and body z (FRD, pointing down) is the negative of that.
	//
	// The vertical component is deliberately gravity alone, matching stock's
	// MPC_ACC_DECOUPLE default in PositionControl::_accelerationControl(). Subtracting
	// acceleration_sp(2) here - as this once did, and as stock does only when that
	// parameter is unset - inverts body_z whenever the commanded downward acceleration
	// exceeds one g. The geometry is not wrong: producing that acceleration really would
	// require thrusting downward. A multirotor cannot, and two callers ask for it
	// routinely. CommandFrontEnd uses a_z = 100 as its pre-takeoff "make no thrust"
	// sentinel, not as a literal request, and coupled that left body_z exactly
	// anti-parallel to world down - straight into limitTilt()'s degenerate branch, which
	// resolves the ambiguity to an arbitrary +x lean of the whole tilt limit. That is a
	// 12 deg forward pitch commanded while still on the pad, which takeoff then rams the
	// vehicle into. The second caller is any firm descent: MC_PD_Z_D alone reaches one g
	// at a 1.2 m/s climb-rate error, inside MPC_Z_VEL_MAX_DN.
	//
	// Decoupled, both cases keep the vehicle level and let the collective saturate low,
	// which is the answer the physics actually supports. The vertical channel is not
	// lost - it reaches the collective through thrust_ned_z below, exactly as stock.
	_body_z_setpoint = Vector3f(-acceleration_sp(0), -acceleration_sp(1), CONSTANTS_ONE_G);

	// Stage 2, magnitude: moved to the shared stage verbatim. This law's version was the
	// one the other laws adopted, so nothing here changes - same derivation, same signed
	// cos(tilt) gated at 0.1, same clamp.
	_thrust_setpoint = _trajectory_stage.computeThrustSetpoint(state, command, acceleration_sp);
}

Vector3f CascadedPdController::attitudeToRateSetpoint(const Quatf &q)
{
	// Stage 3: the geometric attitude error on SO(3),
	//   e_R = 1/2 vee(R_des' R - R' R_des)
	// Dcm::vee() is the inverse of Vector3::hat(), so it is the standard
	// [-S(1,2), S(0,2), -S(0,1)] and matches the prototype's own vee().
	const Dcmf R(q);
	const Dcmf R_des(_attitude_setpoint);
	_attitude_error = 0.5f * Dcmf(R_des.transpose() * R - R.transpose() * R_des).vee();

	// The law is tau = -Kp*e_R - Kd*w, but it is expressed here as a rate setpoint
	// because Kd*(w_sp - w) with w_sp = -(Kp/Kd)*e_R is algebraically identical and
	// costs nothing. What it buys: vehicle_rates_setpoint carries a real signal instead
	// of NaN, and the autotune injection in MulticopterController has somewhere
	// meaningful to land.
	// _att_d(0) and (1) are floored above zero in updateParams(), so this cannot divide
	// by zero.
	Vector3f rate_setpoint{};
	rate_setpoint(0) = -(_att_p / _att_d(0)) * _attitude_error(0);
	rate_setpoint(1) = -(_att_p / _att_d(1)) * _attitude_error(1);

	// Yaw stays zero: damp toward no rotation, never toward a heading. _attitude_error(2)
	// is structurally zero anyway because R_des was built around the current heading, but
	// not commanding it is the part that matters.
	return rate_setpoint;
}

bool CascadedPdController::update(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
				  float dt, mc_ctrl::ControllerOutput &output)
{
	// HARD CONTRACT from the interface. The shared stage's position integrator is the one
	// thing here that genuinely winds up, and zeroing it on the ground is what keeps the
	// vehicle from leaping at takeoff. The cached stage outputs go too, so a mode change
	// cannot carry a stale attitude setpoint across.
	if (command.reset_integrals || !state.armed) {
		_rate_setpoint.setZero();
		_trajectory_stage.resetIntegral();
		_trajectory_stage.invalidateStage();
	}

	switch (command.level) {
	case mc_ctrl::ControlLevel::Trajectory: {
			// Gated on a fresh local position sample so the position stage keeps
			// mc_pos_control's cadence instead of being pulled up to gyro rate.
			if (state.freshness.position_new || !_trajectory_stage.stageValid()) {
				stepTrajectoryToBodyZ(state, command);
			}

			// R_des is rebuilt EVERY cycle, deliberately outside the gate above. The
			// tilt direction it is built from is a position-stage quantity and is
			// correctly held between position samples; the heading it is anchored on
			// is not. Holding the heading too would freeze the whole reference frame
			// for the duration of a position interval while the vehicle keeps
			// rotating - a stale-frame error of r*dt_position that the geometric
			// error then reads as roll and pitch to correct. Rebuilding costs one
			// normalize, one acos and a handful of trig at gyro rate.
			bodyzToAttitudeSetpoint(_body_z_setpoint, state, command);

			_rate_setpoint = attitudeToRateSetpoint(state.q);
			break;
		}

	case mc_ctrl::ControlLevel::Attitude: {
			// Stabilized/Altitude. Keep the pilot's tilt, discard the pilot's heading:
			// rebuilding R_des around state.heading is what makes "no yaw control" hold
			// at this level too, rather than only in Position mode.
			const Vector3f commanded_body_z = Dcmf(command.attitude_sp).col(2);
			bodyzToAttitudeSetpoint(commanded_body_z, state, command);

			_thrust_setpoint = command.thrust_body_sp;
			_rate_setpoint = attitudeToRateSetpoint(state.q);
			_trajectory_stage.invalidateStage();
			break;
		}

	case mc_ctrl::ControlLevel::BodyRate:
		// Acro. No attitude stage to run, so the rate setpoint goes straight into the
		// same damping line below and the D gains do double duty as rate gains.
		_rate_setpoint = command.rate_sp;
		_thrust_setpoint = command.thrust_body_sp;
		_attitude_setpoint = Quatf(NAN, NAN, NAN, NAN);
		_attitude_error.setZero();
		_trajectory_stage.invalidateStage();
		break;

	case mc_ctrl::ControlLevel::None:
	default:
		return false;
	}

	const Vector3f rate_error = (_rate_setpoint + _autotune_rate_sp) - state.angular_velocity;
	Vector3f torque = _att_d.emult(rate_error);

	for (int i = 0; i < 3; i++) {
		torque(i) = PX4_ISFINITE(torque(i)) ? math::constrain(torque(i), -1.f, 1.f) : 0.f;
	}

	// Last line of defence: this class is the framework's failsafe, so it must not be
	// the thing that hands a NaN to the allocator.
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

	(void)dt;	// no integrators, no filters: nothing in this law is rate dependent
	return true;
}

void CascadedPdController::getLocalPositionSetpoint(vehicle_local_position_setpoint_s &sp) const
{
	// Telemetry only. Stock fills this from PositionControl's internal setpoints; here
	// it comes from what the PD stage actually used, so a log still shows what was asked
	// for versus what was flown. The attitude and thrust setpoints are this law's own, so
	// the stage cannot source them itself.
	_trajectory_stage.fillLocalPositionSetpoint(sp, _attitude_setpoint, _thrust_setpoint);
}

void CascadedPdController::fillStatus(mc_controller_status_s &status) const
{
	status.debug[0] = _attitude_error(0);
	status.debug[1] = _attitude_error(1);
	status.debug[2] = _attitude_error(2);
	status.debug[3] = _rate_setpoint(0);
	status.debug[4] = _rate_setpoint(1);
	status.debug[5] = _rate_setpoint(2);
	status.debug[6] = _thrust_setpoint(2);
	status.debug[7] = _trajectory_stage.stageValid() ? 1.f : 0.f;
}
