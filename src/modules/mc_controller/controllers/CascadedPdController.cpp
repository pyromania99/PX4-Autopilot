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

#include "CascadedPidController.hpp"

#include <mathlib/math/Limits.hpp>

using namespace matrix;

CascadedPidController::CascadedPidController(ModuleParams *parent) :
	MulticopterControllerBase(parent)
{
	CascadedPidController::updateParams();
	CascadedPidController::reset();
}

void CascadedPidController::updateParams()
{
	ModuleParams::updateParams();

	// --- position stage (MulticopterPositionControl.cpp:198-204) ------------
	// NOT pushed into _position_control here: this runs on the rate_ctrl queue via the
	// ModuleParams cascade, and _position_control belongs to the outer stage. Flag it
	// instead; applyOuterParams() picks it up on whichever queue runs the position
	// stage. See the CROSS-THREAD CONTRACT in MulticopterControllerBase.hpp.
	_outer_params_dirty.store(true);

	// --- attitude stage (mc_att_control_main.cpp:94-101) --------------------
	_attitude_control.setProportionalGain(
		Vector3f(_param_mc_roll_p.get(), _param_mc_pitch_p.get(), _param_mc_yaw_p.get()),
		_param_mc_yaw_weight.get());
	_attitude_control.setRateLimit(Vector3f(math::radians(_param_mc_rollrate_max.get()),
						math::radians(_param_mc_pitchrate_max.get()),
						math::radians(_param_mc_yawrate_max.get())));

	// --- rate stage (MulticopterRateControl.cpp:82-96) ----------------------
	const Vector3f rate_k = Vector3f(_param_mc_rollrate_k.get(), _param_mc_pitchrate_k.get(),
					 _param_mc_yawrate_k.get());
	_rate_control.setPidGains(
		rate_k.emult(Vector3f(_param_mc_rollrate_p.get(), _param_mc_pitchrate_p.get(), _param_mc_yawrate_p.get())),
		rate_k.emult(Vector3f(_param_mc_rollrate_i.get(), _param_mc_pitchrate_i.get(), _param_mc_yawrate_i.get())),
		rate_k.emult(Vector3f(_param_mc_rollrate_d.get(), _param_mc_pitchrate_d.get(), _param_mc_yawrate_d.get())));
	_rate_control.setIntegratorLimit(
		Vector3f(_param_mc_rr_int_lim.get(), _param_mc_pr_int_lim.get(), _param_mc_yr_int_lim.get()));
	_rate_control.setFeedForwardGain(
		Vector3f(_param_mc_rollrate_ff.get(), _param_mc_pitchrate_ff.get(), _param_mc_yawrate_ff.get()));

	_output_lpf_yaw.setCutoffFreq(_param_mc_yaw_tq_cutoff.get());
}

void CascadedPidController::applyOuterParams()
{
	// Called from whichever queue runs the position stage, never from updateParams().
	bool dirty = true;

	if (!_outer_params_dirty.compare_exchange(&dirty, false)) {
		return;
	}

	_position_control.setPositionGains(Vector3f(_param_mpc_xy_p.get(), _param_mpc_xy_p.get(), _param_mpc_z_p.get()));
	_position_control.setVelocityGains(
		Vector3f(_param_mpc_xy_vel_p_acc.get(), _param_mpc_xy_vel_p_acc.get(), _param_mpc_z_vel_p_acc.get()),
		Vector3f(_param_mpc_xy_vel_i_acc.get(), _param_mpc_xy_vel_i_acc.get(), _param_mpc_z_vel_i_acc.get()),
		Vector3f(_param_mpc_xy_vel_d_acc.get(), _param_mpc_xy_vel_d_acc.get(), _param_mpc_z_vel_d_acc.get()));
	_position_control.setHorizontalThrustMargin(_param_mpc_thr_xy_marg.get());
	_position_control.decoupleHorizontalAndVecticalAcceleration(_param_mpc_acc_decouple.get());
}

void CascadedPidController::reset()
{
	// Inner stage only. _position_control is the outer stage's - see resetOuter().
	_rate_control.resetIntegral();
	_output_lpf_yaw.reset(0.f);
	_rate_setpoint.setZero();
	_thrust_setpoint.setZero();
	_attitude_setpoint = Quatf();
	_autotune_rate_sp.setZero();
	_yaw_sp_move_rate = 0.f;
	_position_stage_valid = false;
}

void CascadedPidController::resetOuter()
{
	_position_control.resetIntegral();
}

bool CascadedPidController::updateOuter(const mc_ctrl::ControllerState &state,
					const mc_ctrl::ControllerCommand &command, float dt,
					vehicle_attitude_setpoint_s &attitude_setpoint)
{
	// Runs on nav_and_controllers at position rate. PositionControl::getAttitudeSetpoint()
	// fills yaw_sp_move_rate from the trajectory yaw rate, and publishing it is how the
	// inner loop receives that feed-forward - exactly as stock does between
	// mc_pos_control and mc_att_control.
	return stepTrajectoryToAttitude(state, command, dt, attitude_setpoint);
}

void CascadedPidController::setAllocatorFeedback(const mc_ctrl::AllocatorFeedback &feedback)
{
	_rate_control.setSaturationStatus(feedback.saturation_positive, feedback.saturation_negative);
}

bool CascadedPidController::stepTrajectoryToAttitude(const mc_ctrl::ControllerState &state,
		const mc_ctrl::ControllerCommand &command, float dt_pos, vehicle_attitude_setpoint_s &attitude_setpoint)
{
	// This is the single place the position stage runs, from either queue, so it is
	// also the only safe place to touch _position_control's parameters and integrator.
	applyOuterParams();

	// HARD CONTRACT from the interface, applied to the outer stage here rather than in
	// update() so it always executes on the queue that owns _position_control.
	if (command.reset_integrals) {
		_position_control.resetIntegral();
	}

	PositionControlStates states;
	states.position = state.position;
	states.velocity = state.velocity;
	states.acceleration = state.acceleration;
	states.yaw = state.heading;

	_position_control.setState(states);
	_position_control.setHoverThrust(state.hover_thrust);
	_position_control.setTiltLimit(command.tilt_limit);
	_position_control.setThrustLimits(command.thrust_min, command.thrust_max);
	_position_control.setVelocityLimits(command.vel_limit_xy, command.vel_limit_up, command.vel_limit_down);

	trajectory_setpoint_s sp{};
	sp.timestamp = command.timestamp;
	command.position_sp.copyTo(sp.position);
	command.velocity_sp.copyTo(sp.velocity);
	command.acceleration_sp.copyTo(sp.acceleration);
	command.jerk_sp.copyTo(sp.jerk);
	sp.yaw = command.yaw_sp;
	sp.yawspeed = command.yawspeed_sp;

	// Horizontal velocity uncontrolled: reset the XY integrators to avoid
	// over-corrections when it resumes (MulticopterPositionControl.cpp:565-570).
	if ((!PX4_ISFINITE(sp.velocity[0]) || !PX4_ISFINITE(sp.velocity[1]))
	    && (!PX4_ISFINITE(sp.position[0]) || !PX4_ISFINITE(sp.position[1]))) {
		_position_control.resetIntegralXY();
	}

	_position_control.setInputSetpoint(sp);

	if (!_position_control.update(dt_pos)) {
		return false;
	}

	_position_control.getAttitudeSetpoint(attitude_setpoint);
	return true;
}

Vector3f CascadedPidController::stepAttitudeToRates(const Quatf &q, const Quatf &q_sp, float yaw_sp_move_rate)
{
	_attitude_control.setAttitudeSetpoint(q_sp, yaw_sp_move_rate);
	return _attitude_control.update(q);
}

void CascadedPidController::runRateStage(const mc_ctrl::ControllerState &state, float dt,
		mc_ctrl::ControllerOutput &output)
{
	Vector3f torque = _rate_control.update(state.angular_velocity, _rate_setpoint, state.angular_accel, dt,
					       state.maybe_landed || state.landed);

	// Low-pass the yaw axis to reduce high frequency torque from rotor acceleration
	// (MulticopterRateControl.cpp:225).
	torque(2) = _output_lpf_yaw.update(torque(2), dt);

	output.torque = torque;
	output.thrust = _thrust_setpoint;
	output.rate_setpoint = _rate_setpoint;
	output.attitude_setpoint = _attitude_setpoint;
	output.valid = true;
}

bool CascadedPidController::update(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
				   float dt, mc_ctrl::ControllerOutput &output)
{
	// HARD CONTRACT from the interface: honour reset_integrals. Only the inner stage's
	// integrator here; the position integrator is reset inside stepTrajectoryToAttitude(),
	// which is the only code that runs on the queue owning _position_control.
	if (command.reset_integrals) {
		_rate_control.resetIntegral();
	}

	// Stock resets the rate integrator whenever disarmed or not a rotary wing
	// (MulticopterRateControl.cpp:182-184).
	if (!state.armed) {
		_rate_control.resetIntegral();
	}

	switch (command.level) {
	case mc_ctrl::ControlLevel::Trajectory: {
			if (command.outer_stage_complete) {
				// Split across work queues: the position stage already ran on
				// nav_and_controllers and its result arrived via the published
				// attitude setpoint. Do not re-run it here.
				_attitude_setpoint = command.attitude_sp;
				_thrust_setpoint = command.thrust_body_sp;
				_yaw_sp_move_rate = command.yaw_sp_move_rate;
				_position_stage_valid = true;

			} else if (state.freshness.position_new || !_position_stage_valid) {
				// Single-queue mode: run the position stage inline, gated on a fresh
				// local position sample so it keeps mc_pos_control's cadence rather
				// than being pulled up to gyro rate.
				vehicle_attitude_setpoint_s attitude_setpoint{};

				if (stepTrajectoryToAttitude(state, command, state.freshness.dt_position, attitude_setpoint)) {
					_attitude_setpoint = Quatf(attitude_setpoint.q_d);
					_thrust_setpoint = Vector3f(attitude_setpoint.thrust_body);
					// Carry the yaw feed-forward through to the attitude stage, exactly as
					// stock does via the published vehicle_attitude_setpoint.
					_yaw_sp_move_rate = PX4_ISFINITE(attitude_setpoint.yaw_sp_move_rate)
							    ? attitude_setpoint.yaw_sp_move_rate : 0.f;
					_position_stage_valid = true;

				} else if (!_position_stage_valid) {
					// Never produced a valid attitude setpoint: refuse rather than
					// command a stale or zero attitude.
					return false;
				}
			}
		}

		// a fresh attitude setpoint feeds the attitude stage
		[[fallthrough]];

	case mc_ctrl::ControlLevel::Attitude: {
			if (command.level == mc_ctrl::ControlLevel::Attitude) {
				_attitude_setpoint = command.attitude_sp;
				_thrust_setpoint = command.thrust_body_sp;
			}

			if (state.freshness.attitude_new || command.level == mc_ctrl::ControlLevel::Attitude) {
				const float yaw_sp_move_rate = (command.level == mc_ctrl::ControlLevel::Attitude)
							       ? command.yaw_sp_move_rate : _yaw_sp_move_rate;
				_rate_setpoint = stepAttitudeToRates(state.q, _attitude_setpoint, yaw_sp_move_rate);

				// Autotune injects an additive rate setpoint (mc_att_control_main.cpp:355).
				_rate_setpoint += _autotune_rate_sp;
			}
		}

		// the rate stage runs every cycle
		[[fallthrough]];

	case mc_ctrl::ControlLevel::BodyRate: {
			if (command.level == mc_ctrl::ControlLevel::BodyRate) {
				_rate_setpoint = command.rate_sp;
				_thrust_setpoint = command.thrust_body_sp;
				_attitude_setpoint = Quatf(NAN, NAN, NAN, NAN);
			}

			runRateStage(state, dt, output);
			break;
		}

	case mc_ctrl::ControlLevel::None:
	default:
		return false;
	}

	return true;
}
