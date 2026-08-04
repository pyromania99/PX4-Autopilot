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
 * @file CascadedPdController.hpp
 * @brief Cascaded PD with a geometric (SO(3)) attitude law. MC_CTRL_ALG=1.
 *
 * Three stages, no integrators anywhere:
 *
 *   1. outer   position/velocity PD              -> desired translational acceleration
 *   2. middle  desired acceleration              -> desired attitude R_des + collective
 *   3. inner   e_R = 1/2 vee(R_des' R - R' R_des) -> body torque
 *
 * YAW IS NOT CONTROLLED. R_des takes its heading from the vehicle's *current*
 * heading every cycle, and the yaw axis gets rate damping only (MC_PD_YAWR_D,
 * default 0). That is deliberate, not an omission: a multirotor that has lost a
 * rotor cannot hold heading, and surrendering yaw is what leaves enough control
 * authority to hold position. The cost is that RC yaw stick and mission yaw
 * setpoints do nothing.
 *
 * This is NOT the stock cascade and makes no claim of equivalence to
 * MC_CTRL_ALG=0. It shares no code with PositionControl / AttitudeControl /
 * RateControl and none of their MPC_ / MC_ gains - only the framework-level
 * limits CommandFrontEnd already computes (tilt, thrust, velocity) and the hover
 * thrust estimate.
 *
 * Beware that this class is also the framework's always-allocated failsafe
 * (MulticopterController::_reference): every input is NaN-guarded and every
 * output clamped, because there is nothing behind it to catch a bad cycle.
 */

#pragma once

#include <MulticopterControllerBase.hpp>

#include <uORB/topics/rate_ctrl_status.h>
#include <uORB/topics/vehicle_local_position_setpoint.h>

class CascadedPdController : public MulticopterControllerBase
{
public:
	explicit CascadedPdController(ModuleParams *parent);
	~CascadedPdController() override = default;

	const char *name() const override { return "cascaded_pd"; }
	uint8_t supportedLevels() const override { return mc_ctrl::kAllLevels; }

	void reset() override;

	bool update(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
		    float dt, mc_ctrl::ControllerOutput &output) override;

	/**
	 * Single work queue. hasOuterStage() exists for trajectory stages too heavy for
	 * the gyro-rate queue; this one is a few dozen flops, so paying for the
	 * cross-queue state-partitioning contract buys nothing. The position stage still
	 * runs at position rate because update() gates it on state.freshness.position_new
	 * and feeds it state.freshness.dt_position.
	 */
	bool hasOuterStage() const override { return false; }

	void fillLocalPositionSetpoint(vehicle_local_position_setpoint_s &sp) const override
	{
		getLocalPositionSetpoint(sp);
	}

	void fillStatus(mc_controller_status_s &status) const override;

	/// Additive rate setpoint from mc_autotune_attitude_control, applied to the
	/// implied rate setpoint the attitude stage produces (see update()).
	void setAutotuneRateSetpoint(const matrix::Vector3f &rate_sp) { _autotune_rate_sp = rate_sp; }
	void clearAutotuneRateSetpoint() { _autotune_rate_sp.setZero(); }

	/// For the module's rate_ctrl_status publication. A PD law has no integrators, so
	/// this reports zeros rather than leaving the topic unpublished - mc_autotune and
	/// the log consumers expect it to keep arriving.
	void getRateControlStatus(rate_ctrl_status_s &status) const
	{
		status.rollspeed_integ = 0.f;
		status.pitchspeed_integ = 0.f;
		status.yawspeed_integ = 0.f;
	}

	/// For the module's vehicle_local_position_setpoint publication.
	void getLocalPositionSetpoint(vehicle_local_position_setpoint_s &sp) const;

protected:
	void updateParams() override;

private:
	/// Stages 1+2: position/velocity PD -> desired acceleration -> R_des + collective.
	/// Writes _attitude_setpoint and _thrust_setpoint.
	void stepTrajectoryToAttitude(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command);

	/// Stage 2, shared by the Trajectory and Attitude levels: a desired body-z
	/// direction plus the *current* heading becomes R_des.
	void bodyzToAttitudeSetpoint(matrix::Vector3f body_z, const mc_ctrl::ControllerState &state,
				     const mc_ctrl::ControllerCommand &command);

	/// Stage 3: attitude error -> implied rate setpoint. The yaw axis is always zero.
	matrix::Vector3f attitudeToRateSetpoint(const matrix::Quatf &q);

	// Cached across cycles: the trajectory stage runs at position rate, the rest at
	// gyro rate.
	matrix::Quatf _attitude_setpoint{};
	matrix::Vector3f _thrust_setpoint{};
	matrix::Vector3f _rate_setpoint{};
	matrix::Vector3f _autotune_rate_sp{};

	/// Telemetry only, for vehicle_local_position_setpoint.
	matrix::Vector3f _position_setpoint{NAN, NAN, NAN};
	matrix::Vector3f _velocity_setpoint{NAN, NAN, NAN};
	matrix::Vector3f _acceleration_setpoint{NAN, NAN, NAN};

	/// Last attitude error, for mc_controller_status.debug[].
	matrix::Vector3f _attitude_error{};

	bool _position_stage_valid{false};

	// Cached gains, so update() never touches the parameter system.
	matrix::Vector3f _pos_p{};	///< [1/s^2] (MC_PD_XY_P, MC_PD_XY_P, MC_PD_Z_P)
	matrix::Vector3f _pos_d{};	///< [1/s]   (MC_PD_XY_D, MC_PD_XY_D, MC_PD_Z_D)
	matrix::Vector3f _att_d{};	///< [s/rad] (MC_PD_ATT_D, MC_PD_ATT_D, MC_PD_YAWR_D)
	float _att_p{0.f};		///< [1/rad] MC_PD_ATT_P, roll/pitch only

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::MC_PD_XY_P>)    _param_mc_pd_xy_p,
		(ParamFloat<px4::params::MC_PD_XY_D>)    _param_mc_pd_xy_d,
		(ParamFloat<px4::params::MC_PD_Z_P>)     _param_mc_pd_z_p,
		(ParamFloat<px4::params::MC_PD_Z_D>)     _param_mc_pd_z_d,
		(ParamFloat<px4::params::MC_PD_ATT_P>)   _param_mc_pd_att_p,
		(ParamFloat<px4::params::MC_PD_ATT_D>)   _param_mc_pd_att_d,
		(ParamFloat<px4::params::MC_PD_YAWR_D>)  _param_mc_pd_yawr_d
	)
};
