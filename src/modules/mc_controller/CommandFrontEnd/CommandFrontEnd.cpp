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

#include "CommandFrontEnd.hpp"

#include <mathlib/math/Limits.hpp>

using namespace matrix;
using namespace time_literals;

/// A setpoint with every field NaN, matching PositionControl::empty_trajectory_setpoint.
static trajectory_setpoint_s emptyTrajectorySetpoint()
{
	trajectory_setpoint_s sp{};

	for (int i = 0; i < 3; i++) {
		sp.position[i] = NAN;
		sp.velocity[i] = NAN;
		sp.acceleration[i] = NAN;
		sp.jerk[i] = NAN;
	}

	sp.yaw = NAN;
	sp.yawspeed = NAN;
	return sp;
}

CommandFrontEnd::CommandFrontEnd(ModuleParams *parent) :
	ModuleParams(parent)
{
	_trajectory_setpoint = emptyTrajectorySetpoint();
	_last_valid_setpoint = emptyTrajectorySetpoint();
	_tilt_limit_slew_rate.setSlewRate(0.2f);   // MulticopterPositionControl.cpp:53
	CommandFrontEnd::updateParams();
}

void CommandFrontEnd::updateParams()
{
	ModuleParams::updateParams();

	_takeoff.setSpoolupTime(_param_com_spoolup_time.get());
	_takeoff.setTakeoffRampTime(_param_mpc_tko_ramp_t.get());
	_takeoff.generateInitialRampValue(_param_mpc_z_vel_p_acc.get());
}

void CommandFrontEnd::reset()
{
	_command = mc_ctrl::ControllerCommand{};
	_previous_level = mc_ctrl::ControlLevel::None;
	_trajectory_setpoint = emptyTrajectorySetpoint();
	_last_valid_setpoint = emptyTrajectorySetpoint();
	_time_position_control_enabled = 0;
	_position_control_was_enabled = false;
	_time_outer_stage_enabled = 0;
	_outer_stage_stale = false;
	_tilt_limit_slew_rate.setForcedValue(math::radians(_param_mpc_tiltmax_lnd.get()));
}

void CommandFrontEnd::setVehicleStatus(const vehicle_status_s &vs)
{
	_vehicle_type = vs.vehicle_type;
	_in_transition = vs.in_transition_mode;
	_is_tailsitter = vs.is_vtol_tailsitter;
	_stick_to_attitude.setVtol(vs.is_vtol);
}

void CommandFrontEnd::applyEkfResetsToSetpoint(const mc_ctrl::EkfResets &resets, trajectory_setpoint_s &sp) const
{
	// Only shift a setpoint that predates the reset; a setpoint generated after the
	// estimator jumped is already expressed in the new frame.
	// (MulticopterPositionControl.cpp:693)
	if (resets.vxy) {
		sp.velocity[0] += resets.delta_vxy(0);
		sp.velocity[1] += resets.delta_vxy(1);
	}

	if (resets.vz) {
		sp.velocity[2] += resets.delta_vz;
	}

	if (resets.xy) {
		sp.position[0] += resets.delta_xy(0);
		sp.position[1] += resets.delta_xy(1);
	}

	if (resets.z) {
		sp.position[2] += resets.delta_z;
	}

	if (resets.heading) {
		sp.yaw = wrap_pi(sp.yaw + resets.delta_heading);
	}
}

trajectory_setpoint_s CommandFrontEnd::generateFailsafeSetpoint(uint64_t now,
		const mc_ctrl::ControllerState &state) const
{
	trajectory_setpoint_s sp = emptyTrajectorySetpoint();
	sp.timestamp = now;

	if (Vector2f(state.velocity).isAllFinite()) {
		// don't move along xy
		sp.velocity[0] = sp.velocity[1] = 0.f;

	} else {
		// descend with land speed since we can't stop
		sp.acceleration[0] = sp.acceleration[1] = 0.f;
		sp.velocity[2] = _param_mpc_land_speed.get();
	}

	if (PX4_ISFINITE(state.velocity(2))) {
		if (!PX4_ISFINITE(sp.velocity[2])) {
			sp.velocity[2] = 0.f;
		}

	} else {
		// emergency descend with a bit below hover thrust
		sp.velocity[2] = NAN;
		sp.acceleration[2] = .3f;
	}

	return sp;
}

void CommandFrontEnd::buildTrajectoryCommand(const mc_ctrl::ControllerState &state, uint64_t now)
{
	// Latch when position control became active, so a stale setpoint from before
	// the mode switch cannot be accepted.
	if (!_position_control_was_enabled) {
		_time_position_control_enabled = now;
		_position_control_was_enabled = true;
	}

	applyEkfResetsToSetpoint(state.resets, _trajectory_setpoint);

	// No fresh setpoint since position control started -> failsafe.
	if (_trajectory_setpoint.timestamp < _time_position_control_enabled) {
		// Accept the last valid setpoint for a short window before giving up
		// (MulticopterPositionControl.cpp:577-601).
		if ((_last_valid_setpoint.timestamp != 0) && (now < _last_valid_setpoint.timestamp + 200_ms)) {
			_trajectory_setpoint = _last_valid_setpoint;

		} else {
			_trajectory_setpoint = generateFailsafeSetpoint(now, state);
		}

	} else {
		_last_valid_setpoint = _trajectory_setpoint;
	}

	// --- constraints and takeoff -------------------------------------------
	if (!PX4_ISFINITE(_vehicle_constraints.speed_up)
	    || (_vehicle_constraints.speed_up > _param_mpc_z_vel_max_up.get())) {
		_vehicle_constraints.speed_up = _param_mpc_z_vel_max_up.get();
	}

	if (_vcm.flag_control_offboard_enabled) {
		const bool want_takeoff = _vcm.flag_armed && (now < _trajectory_setpoint.timestamp + 1_s);

		if (want_takeoff && PX4_ISFINITE(_trajectory_setpoint.position[2])
		    && (_trajectory_setpoint.position[2] < state.position(2))) {
			_vehicle_constraints.want_takeoff = true;

		} else if (want_takeoff && PX4_ISFINITE(_trajectory_setpoint.velocity[2])
			   && (_trajectory_setpoint.velocity[2] < 0.f)) {
			_vehicle_constraints.want_takeoff = true;

		} else if (want_takeoff && PX4_ISFINITE(_trajectory_setpoint.acceleration[2])
			   && (_trajectory_setpoint.acceleration[2] < 0.f)) {
			_vehicle_constraints.want_takeoff = true;

		} else {
			_vehicle_constraints.want_takeoff = false;
		}

		_vehicle_constraints.speed_up = _param_mpc_z_vel_max_up.get();
		_vehicle_constraints.speed_down = _param_mpc_z_vel_max_dn.get();
	}

	const float dt = state.freshness.dt_position;

	_takeoff.updateTakeoffState(_vcm.flag_armed, state.landed, _vehicle_constraints.want_takeoff,
				    _vehicle_constraints.speed_up, _param_com_throw_en.get(), state.timestamp_sample);

	const bool not_taken_off = (_takeoff.getTakeoffState() < TakeoffState::rampup);
	const bool flying = (_takeoff.getTakeoffState() >= TakeoffState::flight);
	const bool flying_but_ground_contact = (flying && state.ground_contact);

	// make sure takeoff ramp is not amended by acceleration feed-forward
	if ((_takeoff.getTakeoffState() == TakeoffState::rampup) && PX4_ISFINITE(_trajectory_setpoint.velocity[2])) {
		_trajectory_setpoint.acceleration[2] = NAN;
	}

	if (not_taken_off || flying_but_ground_contact) {
		// not flying yet: avoid any corrections, and command a strong downward
		// acceleration so no thrust is produced
		_trajectory_setpoint = emptyTrajectorySetpoint();
		_trajectory_setpoint.timestamp = state.timestamp_sample;
		_trajectory_setpoint.acceleration[0] = 0.f;
		_trajectory_setpoint.acceleration[1] = 0.f;
		_trajectory_setpoint.acceleration[2] = 100.f;
		_command.reset_integrals = true;
	}

	// --- limits ------------------------------------------------------------
	const float tilt_limit_deg = (_takeoff.getTakeoffState() < TakeoffState::flight)
				     ? _param_mpc_tiltmax_lnd.get() : _param_mpc_tiltmax_air.get();
	_command.tilt_limit = _tilt_limit_slew_rate.update(math::radians(tilt_limit_deg), dt);

	const float speed_up = _takeoff.updateRamp(dt,
			       PX4_ISFINITE(_vehicle_constraints.speed_up) ? _vehicle_constraints.speed_up
			       : _param_mpc_z_vel_max_up.get());
	const float speed_down = PX4_ISFINITE(_vehicle_constraints.speed_down) ? _vehicle_constraints.speed_down
				 : _param_mpc_z_vel_max_dn.get();

	// Allow ramping from zero thrust on takeoff
	_command.thrust_min = flying ? _param_mpc_thr_min.get() : 0.f;
	_command.thrust_max = _param_mpc_thr_max.get();

	_command.vel_limit_xy = _param_mpc_xy_vel_max.get();
	_command.vel_limit_up = math::min(speed_up, _param_mpc_z_vel_max_up.get());
	_command.vel_limit_down = math::max(speed_down, 0.f);

	// --- publish into the command ------------------------------------------
	_command.position_sp = Vector3f(_trajectory_setpoint.position);
	_command.velocity_sp = Vector3f(_trajectory_setpoint.velocity);
	_command.acceleration_sp = Vector3f(_trajectory_setpoint.acceleration);
	_command.jerk_sp = Vector3f(_trajectory_setpoint.jerk);
	_command.yaw_sp = _trajectory_setpoint.yaw;
	_command.yawspeed_sp = _trajectory_setpoint.yawspeed;
	_command.timestamp = _trajectory_setpoint.timestamp;

	_command.axis_position = _vcm.flag_control_position_enabled;
	_command.axis_velocity = _vcm.flag_control_velocity_enabled;
	_command.axis_altitude = _vcm.flag_control_altitude_enabled;
	_command.axis_climb_rate = _vcm.flag_control_climb_rate_enabled;
	_command.axis_acceleration = _vcm.flag_control_acceleration_enabled;
}

void CommandFrontEnd::buildAttitudeCommand(const mc_ctrl::ControllerState &state, Publications &publications)
{
	if (state.resets.heading || state.resets.quat) {
		_stick_to_attitude.ekfResetHandler(state.resets.delta_heading);
	}

	if (_vcm.flag_control_manual_enabled) {
		// Stabilized / Manual: generate the setpoint from sticks.
		vehicle_attitude_setpoint_s sp{};
		_stick_to_attitude.update(_manual_control_setpoint, state.q, state.unaided_heading,
					  state.freshness.dt_attitude, sp);

		publications.attitude_setpoint = true;
		publications.attitude_setpoint_out = sp;

		_command.attitude_sp = Quatf(sp.q_d);
		_command.yaw_sp_move_rate = sp.yaw_sp_move_rate;
		_command.thrust_body_sp = Vector3f(sp.thrust_body);

	} else {
		// Offboard attitude or VTOL: consume the externally published setpoint.
		// Deliberately NOT republished - that would be a self-feedback path.
		_command.attitude_sp = Quatf(_external_attitude_setpoint.q_d);
		_command.yaw_sp_move_rate = _external_attitude_setpoint.yaw_sp_move_rate;
		_command.thrust_body_sp = Vector3f(_external_attitude_setpoint.thrust_body);
		_command.timestamp = _external_attitude_setpoint.timestamp;
	}

	// Guarantee a finite unit quaternion, as documented in ControllerIO.hpp.
	if (!_command.attitude_sp.isAllFinite() || (fabsf(_command.attitude_sp.norm()) < FLT_EPSILON)) {
		_command.attitude_sp = state.q;
	}

	_command.attitude_sp.normalize();

	for (int i = 0; i < 3; i++) {
		if (!PX4_ISFINITE(_command.thrust_body_sp(i))) {
			_command.thrust_body_sp(i) = 0.f;
		}
	}
}

void CommandFrontEnd::buildBodyRateCommand(const mc_ctrl::ControllerState &state, Publications &publications)
{
	if (_vcm.flag_control_manual_enabled) {
		// Acro
		Vector3f rate_sp{};
		Vector3f thrust_sp{};
		_stick_to_rate.update(_manual_control_setpoint, rate_sp, thrust_sp);

		_command.rate_sp = rate_sp;
		_command.thrust_body_sp = thrust_sp;

		vehicle_rates_setpoint_s sp{};
		sp.roll = rate_sp(0);
		sp.pitch = rate_sp(1);
		sp.yaw = rate_sp(2);
		thrust_sp.copyTo(sp.thrust_body);
		publications.rates_setpoint = true;
		publications.rates_setpoint_out = sp;

	} else {
		// Offboard body_rate. NaN axes fall back to the measured rate, matching
		// MulticopterRateControl.cpp:183-185.
		_command.rate_sp(0) = PX4_ISFINITE(_external_rates_setpoint.roll)
				      ? _external_rates_setpoint.roll  : state.angular_velocity(0);
		_command.rate_sp(1) = PX4_ISFINITE(_external_rates_setpoint.pitch)
				      ? _external_rates_setpoint.pitch : state.angular_velocity(1);
		_command.rate_sp(2) = PX4_ISFINITE(_external_rates_setpoint.yaw)
				      ? _external_rates_setpoint.yaw   : state.angular_velocity(2);
		_command.thrust_body_sp = Vector3f(_external_rates_setpoint.thrust_body);
		_command.timestamp = _external_rates_setpoint.timestamp;
	}

	for (int i = 0; i < 3; i++) {
		if (!PX4_ISFINITE(_command.thrust_body_sp(i))) {
			_command.thrust_body_sp(i) = 0.f;
		}
	}
}

const mc_ctrl::ControllerCommand &CommandFrontEnd::update(const mc_ctrl::ControllerState &state, uint64_t now,
		Publications &publications)
{
	publications = Publications{};

	const mc_ctrl::ControlLevel level =
		mc_ctrl::resolveLevel(_vcm, _vehicle_type, _in_transition, _is_tailsitter);

	// Start from a clean command so no field leaks across levels.
	_command = mc_ctrl::ControllerCommand{};
	_command.level = level;
	_command.timestamp = now;
	_command.manual = _vcm.flag_control_manual_enabled;
	_command.automatic = _vcm.flag_control_auto_enabled;
	_command.offboard = _vcm.flag_control_offboard_enabled;

	// Integrators must be reset on any level change, on the ground, and when
	// disarmed. This is the one contract the controller must honour.
	_command.reset_integrals = (level != _previous_level) || !_vcm.flag_armed || state.landed;

	if (level != mc_ctrl::ControlLevel::Trajectory) {
		// Leaving position control: re-arm the "no setpoint yet" latch so a stale
		// setpoint cannot be accepted when we come back.
		_position_control_was_enabled = false;
	}

	if (!_vcm.flag_control_manual_enabled || (level != mc_ctrl::ControlLevel::Attitude)) {
		// Not generating attitude from sticks: drop the held yaw setpoint so
		// re-entry does not resume a stale one (mc_att_control_main.cpp:305-308).
		_stick_to_attitude.reset(state.q, state.unaided_heading);
	}

	// Latch when the outer stage took over, so its first setpoint gets a grace window
	// instead of being judged stale before it has ever been published.
	if (!_outer_stage_active || (level != mc_ctrl::ControlLevel::Trajectory)) {
		_time_outer_stage_enabled = 0;
		_outer_stage_stale = false;

	} else if (_time_outer_stage_enabled == 0) {
		_time_outer_stage_enabled = now;
	}

	switch (level) {
	case mc_ctrl::ControlLevel::Trajectory:
		if (_outer_stage_active) {
			// A separate outer-loop work item ran the trajectory front end and the
			// controller's outer stage at position rate. Its result arrives here as
			// the published attitude setpoint, exactly as mc_att_control receives
			// mc_pos_control's output in stock PX4.
			_command.attitude_sp = Quatf(_external_attitude_setpoint.q_d);
			_command.yaw_sp_move_rate = _external_attitude_setpoint.yaw_sp_move_rate;
			_command.thrust_body_sp = Vector3f(_external_attitude_setpoint.thrust_body);
			_command.outer_stage_complete = true;

			// The outer item publishes nothing when its stage fails, so a setpoint that
			// stops arriving is the only symptom. Nothing downstream can see it: the
			// inner controller keeps producing a perfectly valid output from the held
			// setpoint, so neither the NaN guard nor the output watchdog ever trips and
			// the vehicle would fly the last good attitude indefinitely.
			const uint64_t newest = math::max(_external_attitude_setpoint.timestamp, _time_outer_stage_enabled);
			_outer_stage_stale = (now > newest + kOuterStageTimeoutUs);

			if (!_command.attitude_sp.isAllFinite() || (fabsf(_command.attitude_sp.norm()) < FLT_EPSILON)) {
				_command.attitude_sp = state.q;
			}

			_command.attitude_sp.normalize();

			for (int i = 0; i < 3; i++) {
				if (!PX4_ISFINITE(_command.thrust_body_sp(i))) {
					_command.thrust_body_sp(i) = 0.f;
				}
			}

			if (!PX4_ISFINITE(_command.yaw_sp_move_rate)) {
				_command.yaw_sp_move_rate = 0.f;
			}

		} else {
			buildTrajectoryCommand(state, now);
		}

		break;

	case mc_ctrl::ControlLevel::Attitude:
		buildAttitudeCommand(state, publications);
		break;

	case mc_ctrl::ControlLevel::BodyRate:
		buildBodyRateCommand(state, publications);
		break;

	case mc_ctrl::ControlLevel::None:
	default:
		break;
	}

	// Derived from the state rather than pushed in by the module. Stock feeds the
	// hover-thrust estimate to two separate consumers - the position controller and
	// this throttle curve (mc_att_control_main.cpp:119-124) - and the framework
	// originally wired only the first, so Stabilized flew on the MPC_THR_HOVER
	// parameter instead of the live estimate: a 21% collective error.
	//
	// Taking it off the state we already receive every cycle means there is no
	// second call site to forget, and a unit test driving update() covers it. A
	// mutation test proved the alternative untestable: with the value pushed in by
	// the module, deleting that call broke flight while every unit test still
	// passed, because no CommandFrontEnd test can observe a missing call in module
	// glue.
	//
	// NAN when invalid, not the parameter value: the mapping keeps its own slew and
	// fallback, matching stock.
	_stick_to_attitude.setHoverThrustEstimate(state.hover_thrust_valid ? state.hover_thrust : NAN);
	_stick_to_attitude.updateSlewRates(state.landed, state.spooled_up, state.freshness.dt_attitude);

	_previous_level = level;
	return _command;
}
