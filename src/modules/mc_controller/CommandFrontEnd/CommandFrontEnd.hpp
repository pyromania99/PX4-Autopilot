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
 * @file CommandFrontEnd.hpp
 * @brief Normalise RC, mission and offboard input into a single ControllerCommand.
 *
 * This is what lets one control law serve every flight mode: whatever the mode
 * commands - a trajectory, an attitude, or body rates - the controller receives
 * a ControllerCommand tagged with the level it was commanded at.
 *
 * The safety-critical setpoint policy lives here rather than in the controller:
 * takeoff ramp, failsafe fallback, EKF-reset application and the limit envelope.
 * A research controller should not be able to get those wrong.
 *
 * Ported from MulticopterPositionControl::Run() (the Trajectory path),
 * mc_att_control_main.cpp (Attitude) and MulticopterRateControl.cpp (BodyRate).
 */

#pragma once

#include "ControlLevelResolver.hpp"

#include <ControllerIO.hpp>
#include <Takeoff.hpp>
#include <lib/mc_manual_mapping/StickToAttitudeSetpoint.hpp>
#include <lib/mc_manual_mapping/StickToRateSetpoint.hpp>
#include <lib/slew_rate/SlewRate.hpp>
#include <px4_platform_common/module_params.h>

#include <uORB/topics/manual_control_setpoint.h>
#include <uORB/topics/trajectory_setpoint.h>
#include <uORB/topics/vehicle_attitude_setpoint.h>
#include <uORB/topics/vehicle_constraints.h>
#include <uORB/topics/vehicle_control_mode.h>
#include <uORB/topics/vehicle_rates_setpoint.h>
#include <uORB/topics/vehicle_status.h>

class CommandFrontEnd : public ModuleParams
{
public:
	explicit CommandFrontEnd(ModuleParams *parent);
	~CommandFrontEnd() override = default;

	/// What the front-end wants the module to publish this cycle.
	struct Publications {
		bool attitude_setpoint{false};	///< publish `attitude_setpoint_out` (we generated it from sticks)
		bool rates_setpoint{false};	///< publish `rates_setpoint_out` (we generated it from sticks)
		vehicle_attitude_setpoint_s attitude_setpoint_out{};
		vehicle_rates_setpoint_s rates_setpoint_out{};
	};

	void reset();

	/**
	 * Inner-loop instance only: tell the front end that a separate outer-loop work
	 * item owns the Trajectory path.
	 *
	 * When active, this instance does NOT run the trajectory front end. At
	 * Trajectory level it instead consumes the vehicle_attitude_setpoint published
	 * by the outer loop and marks the command `outer_stage_complete`, so the
	 * controller skips its own position stage. This is exactly the interface stock
	 * uses between mc_pos_control and mc_att_control.
	 */
	void setOuterStageActive(bool active) { _outer_stage_active = active; }
	bool outerStageActive() const { return _outer_stage_active; }

	/**
	 * True when the outer loop has stopped delivering attitude setpoints at Trajectory
	 * level. The module latches the reference fallback on this: no other guard can see
	 * it, because the inner controller happily keeps tracking the last held setpoint.
	 * Always false unless setOuterStageActive(true) and the level is Trajectory.
	 */
	bool outerStageStale() const { return _outer_stage_stale; }

	// --- inputs, fed by the module -----------------------------------------
	void setControlMode(const vehicle_control_mode_s &vcm) { _vcm = vcm; }
	void setVehicleStatus(const vehicle_status_s &vs);
	void setTrajectorySetpoint(const trajectory_setpoint_s &sp) { _trajectory_setpoint = sp; }
	void setVehicleConstraints(const vehicle_constraints_s &c) { _vehicle_constraints = c; }
	void setManualControlSetpoint(const manual_control_setpoint_s &m) { _manual_control_setpoint = m; }
	// Deliberately NO setHoverThrustEstimate() here. The manual throttle curve needs
	// the hover-thrust estimate, but update() already receives it on the state, so it
	// is read from there - see the comment at the call site in update(). An explicit
	// setter would be a second call site the module has to remember, which is exactly
	// the defect that shipped once and which no unit test could catch.

	void setExternalAttitudeSetpoint(const vehicle_attitude_setpoint_s &sp) { _external_attitude_setpoint = sp; }
	void setExternalRatesSetpoint(const vehicle_rates_setpoint_s &sp) { _external_rates_setpoint = sp; }

	/**
	 * Build the command for this cycle.
	 *
	 * @param state          from VehicleStateProvider (EKF reset deltas still pending)
	 * @param now            hrt timestamp
	 * @param publications   filled with anything the module must publish
	 */
	const mc_ctrl::ControllerCommand &update(const mc_ctrl::ControllerState &state, uint64_t now,
			Publications &publications);

	mc_ctrl::ControlLevel level() const { return _command.level; }

	/// Takeoff state machine output, for the module's takeoff_status publication.
	/// Non-const: TakeoffHandling::getTakeoffState() is not a const method.
	TakeoffState takeoffState() { return _takeoff.getTakeoffState(); }

protected:
	void updateParams() override;

private:
	void applyEkfResetsToSetpoint(const mc_ctrl::EkfResets &resets, trajectory_setpoint_s &sp) const;
	trajectory_setpoint_s generateFailsafeSetpoint(uint64_t now, const mc_ctrl::ControllerState &state) const;
	void buildTrajectoryCommand(const mc_ctrl::ControllerState &state, uint64_t now);
	void buildAttitudeCommand(const mc_ctrl::ControllerState &state, Publications &publications);
	void buildBodyRateCommand(const mc_ctrl::ControllerState &state, Publications &publications);

	mc_ctrl::ControllerCommand _command{};
	mc_ctrl::ControlLevel _previous_level{mc_ctrl::ControlLevel::None};

	vehicle_control_mode_s _vcm{};
	uint8_t _vehicle_type{vehicle_status_s::VEHICLE_TYPE_ROTARY_WING};
	bool _in_transition{false};
	bool _is_tailsitter{false};

	trajectory_setpoint_s _trajectory_setpoint{};
	trajectory_setpoint_s _last_valid_setpoint{};
	vehicle_constraints_s _vehicle_constraints{};
	manual_control_setpoint_s _manual_control_setpoint{};
	vehicle_attitude_setpoint_s _external_attitude_setpoint{};
	vehicle_rates_setpoint_s _external_rates_setpoint{};

	TakeoffHandling _takeoff{};
	SlewRate<float> _tilt_limit_slew_rate{};

	StickToAttitudeSetpoint _stick_to_attitude{this};
	StickToRateSetpoint _stick_to_rate{this};

	/// Grace window for the outer loop's attitude setpoint. Generous relative to the
	/// ~100 Hz position rate it is published at, so a couple of missed cycles are not
	/// treated as a failure, but short enough to catch a stopped outer stage well
	/// inside a second.
	static constexpr uint64_t kOuterStageTimeoutUs{100000}; // 100 ms

	uint64_t _time_position_control_enabled{0};
	bool _position_control_was_enabled{false};
	bool _outer_stage_active{false};
	uint64_t _time_outer_stage_enabled{0};
	bool _outer_stage_stale{false};

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::MPC_XY_VEL_MAX>)    _param_mpc_xy_vel_max,
		(ParamFloat<px4::params::MPC_Z_VEL_MAX_UP>)  _param_mpc_z_vel_max_up,
		(ParamFloat<px4::params::MPC_Z_VEL_MAX_DN>)  _param_mpc_z_vel_max_dn,
		(ParamFloat<px4::params::MPC_LAND_SPEED>)    _param_mpc_land_speed,
		(ParamFloat<px4::params::MPC_THR_MIN>)       _param_mpc_thr_min,
		(ParamFloat<px4::params::MPC_THR_MAX>)       _param_mpc_thr_max,
		(ParamFloat<px4::params::MPC_TILTMAX_AIR>)   _param_mpc_tiltmax_air,
		(ParamFloat<px4::params::MPC_TILTMAX_LND>)   _param_mpc_tiltmax_lnd,
		(ParamFloat<px4::params::MPC_TKO_RAMP_T>)    _param_mpc_tko_ramp_t,
		(ParamFloat<px4::params::MPC_Z_VEL_P_ACC>)   _param_mpc_z_vel_p_acc,
		(ParamFloat<px4::params::COM_SPOOLUP_TIME>)  _param_com_spoolup_time,
		(ParamBool<px4::params::COM_THROW_EN>)       _param_com_throw_en
	)
};
