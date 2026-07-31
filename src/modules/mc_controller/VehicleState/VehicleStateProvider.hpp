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
 * @file VehicleStateProvider.hpp
 * @brief Assemble mc_ctrl::ControllerState from the estimator topics.
 *
 * The velocity filter chain and the EKF-reset bookkeeping are ported from
 * MulticopterPositionControl::set_vehicle_states() and ::adjustSetpointForEKFResets()
 * so every controller sees exactly the state the stock cascade sees.
 *
 * This class owns no uORB subscriptions: the module feeds it messages. That keeps
 * it unit-testable without the uORB runtime.
 */

#pragma once

#include <ControllerIO.hpp>

#include <lib/mathlib/math/filter/AlphaFilter.hpp>
#include <lib/mathlib/math/filter/NotchFilter.hpp>
#include <lib/mathlib/math/WelfordMean.hpp>
#include <px4_platform_common/module_params.h>

#include <uORB/topics/vehicle_angular_velocity.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_land_detected.h>
#include <uORB/topics/vehicle_local_position.h>

class VehicleStateProvider : public ModuleParams
{
public:
	explicit VehicleStateProvider(ModuleParams *parent);
	~VehicleStateProvider() override = default;

	/// Drop all filter and reset state. Call when the controller is (re)selected.
	void reset();

	/**
	 * Feed a new vehicle_local_position sample.
	 * Runs the notch/low-pass chain and latches any EKF reset deltas.
	 */
	void updateLocalPosition(const vehicle_local_position_s &local_position);

	/// Feed a new vehicle_attitude sample. Latches quaternion reset deltas.
	void updateAttitude(const vehicle_attitude_s &attitude);

	/// Feed a new vehicle_angular_velocity sample. This drives the control cycle.
	void updateAngularVelocity(const vehicle_angular_velocity_s &angular_velocity);

	/**
	 * Set the cycle timestamp explicitly.
	 *
	 * Only the outer-loop instance needs this: it is driven by vehicle_local_position
	 * rather than the gyro, so updateAngularVelocity() is never called on it and
	 * timestamp_sample would otherwise stay 0. A zero timestamp silently stalls the
	 * takeoff state machine, which reads it as "now".
	 */
	void setTimestampSample(uint64_t timestamp_sample) { _state.timestamp_sample = timestamp_sample; }

	void updateLandDetected(const vehicle_land_detected_s &land_detected);

	void setArmed(bool armed, bool spooled_up);

	/// @param hover_thrust valid estimate, or NAN to fall back to the parameter value
	void setHoverThrustEstimate(float hover_thrust);

	/// The state for this control cycle.
	const mc_ctrl::ControllerState &getState();

	/// Reset deltas that the command front-end must apply to its setpoints.
	const mc_ctrl::EkfResets &pendingResets() const { return _state.resets; }

	/**
	 * Clear the one-shot signals: EKF reset deltas and the *_new freshness flags.
	 *
	 * Must be called by the module AFTER the controller has consumed the state.
	 * Deliberately not folded into getState(): that returns a reference, and
	 * clearing at the start of the cycle instead would wipe resets latched
	 * earlier in the same cycle by updateLocalPosition().
	 */
	void endCycle();

protected:
	void updateParams() override;

private:
	void configureFilters();

	mc_ctrl::ControllerState _state{};

	// Filter chain, mirroring MulticopterPositionControl.
	AlphaFilter<matrix::Vector2f> _vel_xy_lp_filter{};
	AlphaFilter<float> _vel_z_lp_filter{};
	math::NotchFilter<matrix::Vector2f> _vel_xy_notch_filter{};
	math::NotchFilter<float> _vel_z_notch_filter{};
	AlphaFilter<matrix::Vector2f> _vel_deriv_xy_lp_filter{};
	AlphaFilter<float> _vel_deriv_z_lp_filter{};

	math::WelfordMean<float> _sample_interval_s{};

	// Timestamps for the per-stage dt values.
	uint64_t _last_position_sample{0};
	uint64_t _last_attitude_sample{0};
	uint64_t _last_cycle_sample{0};

	// EKF reset counters.
	uint8_t _xy_reset_counter{0};
	uint8_t _z_reset_counter{0};
	uint8_t _vxy_reset_counter{0};
	uint8_t _vz_reset_counter{0};
	uint8_t _heading_reset_counter{0};
	uint8_t _quat_reset_counter{0};
	/// A fresh subscription is not a reset: the first sample only latches the counters.
	/// Tracked separately for position and attitude because the two topics arrive
	/// independently and either can deliver its first sample first.
	bool _reset_counters_initialised{false};
	bool _quat_reset_counter_initialised{false};

	/// Sample rate the filters were last configured for, so a drifting position rate
	/// re-derives the coefficients instead of keeping the constructor's assumption.
	float _configured_sample_interval_s{NAN};

	float _hover_thrust_estimate{NAN};

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::MPC_VEL_NF_FRQ>) _param_mpc_vel_nf_frq,
		(ParamFloat<px4::params::MPC_VEL_NF_BW>)  _param_mpc_vel_nf_bw,
		(ParamFloat<px4::params::MPC_VEL_LP>)     _param_mpc_vel_lp,
		(ParamFloat<px4::params::MPC_VELD_LP>)    _param_mpc_veld_lp,
		(ParamFloat<px4::params::MPC_THR_HOVER>)  _param_mpc_thr_hover
	)
};
