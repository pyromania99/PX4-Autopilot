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
 * @file CascadedPidController.hpp
 * @brief The stock PX4 cascade, expressed on the pluggable controller interface.
 *
 * This is the reference implementation and the A/B baseline: it wraps the very
 * same PositionControl, AttitudeControl and RateControl objects the stock modules
 * use, and reuses their MPC_ and MC_ parameters. MC_CTRL_ALG=1 should therefore be
 * behaviourally indistinguishable from MC_CTRL_ALG=0.
 *
 * Stage-rate parity matters: the three stock stages run at different rates on two
 * work queues. Each stage here is gated on state.freshness and fed its own dt, or
 * the reference would drift from stock in a way that looks like a controller bug.
 */

#pragma once

#include <AttitudeControl.hpp>
#include <MulticopterControllerBase.hpp>
#include <PositionControl.hpp>
#include <lib/mathlib/math/filter/AlphaFilter.hpp>
#include <lib/rate_control/rate_control.hpp>
#include <px4_platform_common/atomic.h>

#include <uORB/topics/rate_ctrl_status.h>
#include <uORB/topics/vehicle_local_position_setpoint.h>

class CascadedPidController : public MulticopterControllerBase
{
public:
	explicit CascadedPidController(ModuleParams *parent);
	~CascadedPidController() override = default;

	const char *name() const override { return "cascaded_pid"; }
	uint8_t supportedLevels() const override { return mc_ctrl::kAllLevels; }

	void reset() override;
	void resetOuter() override;
	void updateOuterParams() override { applyOuterParams(); }

	bool update(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
		    float dt, mc_ctrl::ControllerOutput &output) override;

	void setAllocatorFeedback(const mc_ctrl::AllocatorFeedback &feedback) override;

	/**
	 * The cascade decomposes cleanly, so it takes stock's work-queue separation:
	 * the position stage runs on nav_and_controllers at position rate and hands the
	 * attitude setpoint down over uORB.
	 *
	 * Cross-thread state is disjoint by construction: _position_control is reached
	 * only from stepTrajectoryToAttitude() and resetOuter(); update() and reset()
	 * touch only _attitude_control, _rate_control and _output_lpf_yaw. Parameter
	 * writes go the same way - updateParams() only raises _outer_params_dirty and
	 * applyOuterParams() does the actual push. Everything crossing between the
	 * stages travels through the published vehicle_attitude_setpoint.
	 */
	bool hasOuterStage() const override { return true; }

	bool updateOuter(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
			 float dt, vehicle_attitude_setpoint_s &attitude_setpoint) override;

	/// Additive rate setpoint from mc_autotune_attitude_control, applied at the
	/// attitude stage exactly as mc_att_control does (mc_att_control_main.cpp:346-357).
	void setAutotuneRateSetpoint(const matrix::Vector3f &rate_sp) { _autotune_rate_sp = rate_sp; }
	void clearAutotuneRateSetpoint() { _autotune_rate_sp.setZero(); }

	/// For the module's rate_ctrl_status publication.
	void getRateControlStatus(rate_ctrl_status_s &status) { _rate_control.getRateControlStatus(status); }

	/// For the module's vehicle_local_position_setpoint publication.
	void getLocalPositionSetpoint(vehicle_local_position_setpoint_s &sp) const
	{
		_position_control.getLocalPositionSetpoint(sp);
	}

	void fillLocalPositionSetpoint(vehicle_local_position_setpoint_s &sp) const override
	{
		_position_control.getLocalPositionSetpoint(sp);
	}

	/// Exposed so the framework can later run partial stacks for controllers that
	/// only implement an inner loop (deferred Stage 2 work in the plan).
	bool stepTrajectoryToAttitude(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
				      float dt_pos, vehicle_attitude_setpoint_s &attitude_setpoint);
	matrix::Vector3f stepAttitudeToRates(const matrix::Quatf &q, const matrix::Quatf &q_sp, float yaw_sp_move_rate);

protected:
	void updateParams() override;


private:
	void runRateStage(const mc_ctrl::ControllerState &state, float dt, mc_ctrl::ControllerOutput &output);

	/// Push cached position-stage parameters into _position_control if they changed.
	/// Only ever called from the queue that runs the position stage.
	void applyOuterParams();

	PositionControl _position_control;
	AttitudeControl _attitude_control;
	RateControl _rate_control;

	AlphaFilter<float> _output_lpf_yaw;	///< MC_YAW_TQ_CUTOFF, applied to yaw torque

	// Cached between stages, since the stages run at different rates.
	matrix::Vector3f _rate_setpoint{};
	matrix::Vector3f _thrust_setpoint{};
	matrix::Quatf _attitude_setpoint{};
	matrix::Vector3f _autotune_rate_sp{};

	/// Yaw feed-forward produced by the position stage
	/// (PositionControl.cpp:272 sets attitude_setpoint.yaw_sp_move_rate = _yawspeed_sp).
	/// Stock forwards this to AttitudeControl via the published attitude setpoint;
	/// dropping it loses trajectory yaw feed-forward entirely.
	float _yaw_sp_move_rate{0.f};

	bool _position_stage_valid{false};

	/// Raised by updateParams() on the rate_ctrl queue, consumed by applyOuterParams()
	/// on the queue that owns _position_control.
	px4::atomic_bool _outer_params_dirty{true};

	DEFINE_PARAMETERS(
		// position stage
		(ParamFloat<px4::params::MPC_XY_P>)          _param_mpc_xy_p,
		(ParamFloat<px4::params::MPC_Z_P>)           _param_mpc_z_p,
		(ParamFloat<px4::params::MPC_XY_VEL_P_ACC>)  _param_mpc_xy_vel_p_acc,
		(ParamFloat<px4::params::MPC_XY_VEL_I_ACC>)  _param_mpc_xy_vel_i_acc,
		(ParamFloat<px4::params::MPC_XY_VEL_D_ACC>)  _param_mpc_xy_vel_d_acc,
		(ParamFloat<px4::params::MPC_Z_VEL_P_ACC>)   _param_mpc_z_vel_p_acc,
		(ParamFloat<px4::params::MPC_Z_VEL_I_ACC>)   _param_mpc_z_vel_i_acc,
		(ParamFloat<px4::params::MPC_Z_VEL_D_ACC>)   _param_mpc_z_vel_d_acc,
		(ParamFloat<px4::params::MPC_THR_XY_MARG>)   _param_mpc_thr_xy_marg,
		(ParamBool<px4::params::MPC_ACC_DECOUPLE>)   _param_mpc_acc_decouple,

		// attitude stage
		(ParamFloat<px4::params::MC_ROLL_P>)         _param_mc_roll_p,
		(ParamFloat<px4::params::MC_PITCH_P>)        _param_mc_pitch_p,
		(ParamFloat<px4::params::MC_YAW_P>)          _param_mc_yaw_p,
		(ParamFloat<px4::params::MC_YAW_WEIGHT>)     _param_mc_yaw_weight,
		(ParamFloat<px4::params::MC_ROLLRATE_MAX>)   _param_mc_rollrate_max,
		(ParamFloat<px4::params::MC_PITCHRATE_MAX>)  _param_mc_pitchrate_max,
		(ParamFloat<px4::params::MC_YAWRATE_MAX>)    _param_mc_yawrate_max,

		// rate stage
		(ParamFloat<px4::params::MC_ROLLRATE_P>)     _param_mc_rollrate_p,
		(ParamFloat<px4::params::MC_ROLLRATE_I>)     _param_mc_rollrate_i,
		(ParamFloat<px4::params::MC_RR_INT_LIM>)     _param_mc_rr_int_lim,
		(ParamFloat<px4::params::MC_ROLLRATE_D>)     _param_mc_rollrate_d,
		(ParamFloat<px4::params::MC_ROLLRATE_FF>)    _param_mc_rollrate_ff,
		(ParamFloat<px4::params::MC_ROLLRATE_K>)     _param_mc_rollrate_k,
		(ParamFloat<px4::params::MC_PITCHRATE_P>)    _param_mc_pitchrate_p,
		(ParamFloat<px4::params::MC_PITCHRATE_I>)    _param_mc_pitchrate_i,
		(ParamFloat<px4::params::MC_PR_INT_LIM>)     _param_mc_pr_int_lim,
		(ParamFloat<px4::params::MC_PITCHRATE_D>)    _param_mc_pitchrate_d,
		(ParamFloat<px4::params::MC_PITCHRATE_FF>)   _param_mc_pitchrate_ff,
		(ParamFloat<px4::params::MC_PITCHRATE_K>)    _param_mc_pitchrate_k,
		(ParamFloat<px4::params::MC_YAWRATE_P>)      _param_mc_yawrate_p,
		(ParamFloat<px4::params::MC_YAWRATE_I>)      _param_mc_yawrate_i,
		(ParamFloat<px4::params::MC_YR_INT_LIM>)     _param_mc_yr_int_lim,
		(ParamFloat<px4::params::MC_YAWRATE_D>)      _param_mc_yawrate_d,
		(ParamFloat<px4::params::MC_YAWRATE_FF>)     _param_mc_yawrate_ff,
		(ParamFloat<px4::params::MC_YAWRATE_K>)      _param_mc_yawrate_k,
		(ParamFloat<px4::params::MC_YAW_TQ_CUTOFF>)  _param_mc_yaw_tq_cutoff
	)
};
