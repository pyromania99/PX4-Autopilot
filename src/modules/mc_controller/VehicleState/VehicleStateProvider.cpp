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

#include "VehicleStateProvider.hpp"

#include <mathlib/math/Limits.hpp>

using namespace matrix;

// dt clamps, matching the stock modules exactly.
static constexpr float kDtPositionMin = 0.002f;		// MulticopterPositionControl.cpp:398
static constexpr float kDtPositionMax = 0.04f;
static constexpr float kDtAttitudeMin = 0.0002f;	// mc_att_control_main.cpp:249
static constexpr float kDtAttitudeMax = 0.02f;
static constexpr float kDtCycleMin = 0.000125f;		// MulticopterRateControl.cpp:128
static constexpr float kDtCycleMax = 0.02f;

VehicleStateProvider::VehicleStateProvider(ModuleParams *parent) :
	ModuleParams(parent)
{
	_sample_interval_s.update(0.01f); // 100 Hz default, as in mc_pos_control
	VehicleStateProvider::updateParams();
}

void VehicleStateProvider::updateParams()
{
	ModuleParams::updateParams();
	configureFilters();
}

void VehicleStateProvider::configureFilters()
{
	const float sample_interval_s = _sample_interval_s.mean();
	_configured_sample_interval_s = sample_interval_s;
	const float sample_freq_hz = 1.f / sample_interval_s;

	// velocity notch filter
	if ((_param_mpc_vel_nf_frq.get() > 0.f) && (_param_mpc_vel_nf_bw.get() > 0.f)) {
		_vel_xy_notch_filter.setParameters(sample_freq_hz, _param_mpc_vel_nf_frq.get(), _param_mpc_vel_nf_bw.get());
		_vel_z_notch_filter.setParameters(sample_freq_hz, _param_mpc_vel_nf_frq.get(), _param_mpc_vel_nf_bw.get());

	} else {
		_vel_xy_notch_filter.disable();
		_vel_z_notch_filter.disable();
	}

	// velocity xy/z low pass filter
	if (_param_mpc_vel_lp.get() > 0.f) {
		_vel_xy_lp_filter.setCutoffFreq(sample_freq_hz, _param_mpc_vel_lp.get());
		_vel_z_lp_filter.setCutoffFreq(sample_freq_hz, _param_mpc_vel_lp.get());

	} else {
		_vel_xy_lp_filter.setAlpha(1.f);
		_vel_z_lp_filter.setAlpha(1.f);
	}

	// velocity derivative xy/z low pass filter
	if (_param_mpc_veld_lp.get() > 0.f) {
		_vel_deriv_xy_lp_filter.setCutoffFreq(sample_freq_hz, _param_mpc_veld_lp.get());
		_vel_deriv_z_lp_filter.setCutoffFreq(sample_freq_hz, _param_mpc_veld_lp.get());

	} else {
		_vel_deriv_xy_lp_filter.setAlpha(1.f);
		_vel_deriv_z_lp_filter.setAlpha(1.f);
	}
}

void VehicleStateProvider::reset()
{
	_state = mc_ctrl::ControllerState{};

	_vel_xy_lp_filter.reset({});
	_vel_z_lp_filter.reset({});
	_vel_xy_notch_filter.reset();
	_vel_z_notch_filter.reset();
	_vel_deriv_xy_lp_filter.reset({});
	_vel_deriv_z_lp_filter.reset({});

	_last_position_sample = 0;
	_last_attitude_sample = 0;
	_last_cycle_sample = 0;
	_reset_counters_initialised = false;
	_quat_reset_counter_initialised = false;
	_hover_thrust_estimate = NAN;
}

void VehicleStateProvider::updateLocalPosition(const vehicle_local_position_s &local_position)
{
	const float dt_s = (_last_position_sample == 0)
			   ? _sample_interval_s.mean()
			   : math::constrain((local_position.timestamp_sample - _last_position_sample) * 1e-6f,
					     kDtPositionMin, kDtPositionMax);
	_last_position_sample = local_position.timestamp_sample;
	_sample_interval_s.update(dt_s);

	// The filter coefficients are derived from the sample rate. Re-derive them when the
	// running mean has drifted away from what they were built for, otherwise they keep
	// the constructor's 100 Hz assumption for the lifetime of the module.
	if (!PX4_ISFINITE(_configured_sample_interval_s)
	    || (fabsf(_sample_interval_s.mean() - _configured_sample_interval_s) > 0.1f * _configured_sample_interval_s)) {
		configureFilters();
	}

	_state.freshness.dt_position = dt_s;
	_state.freshness.position_new = true;

	// --- EKF reset bookkeeping ---------------------------------------------
	// First sample only latches the counters; a fresh subscription is not a reset.
	if (!_reset_counters_initialised) {
		_xy_reset_counter = local_position.xy_reset_counter;
		_z_reset_counter = local_position.z_reset_counter;
		_vxy_reset_counter = local_position.vxy_reset_counter;
		_vz_reset_counter = local_position.vz_reset_counter;
		_heading_reset_counter = local_position.heading_reset_counter;
		_reset_counters_initialised = true;

	} else {
		if (local_position.xy_reset_counter != _xy_reset_counter) {
			_state.resets.xy = true;
			_state.resets.delta_xy = Vector2f(local_position.delta_xy);
			_xy_reset_counter = local_position.xy_reset_counter;
		}

		if (local_position.z_reset_counter != _z_reset_counter) {
			_state.resets.z = true;
			_state.resets.delta_z = local_position.delta_z;
			_z_reset_counter = local_position.z_reset_counter;
		}

		if (local_position.vxy_reset_counter != _vxy_reset_counter) {
			_state.resets.vxy = true;
			_state.resets.delta_vxy = Vector2f(local_position.delta_vxy);
			// Carry the filter state across the discontinuity so the derivative does
			// not spike (MulticopterPositionControl.cpp:717-720).
			_vel_xy_lp_filter.reset(_vel_xy_lp_filter.getState() + Vector2f(local_position.delta_vxy));
			_vel_xy_notch_filter.reset();
			_vxy_reset_counter = local_position.vxy_reset_counter;
		}

		if (local_position.vz_reset_counter != _vz_reset_counter) {
			_state.resets.vz = true;
			_state.resets.delta_vz = local_position.delta_vz;
			_vel_z_lp_filter.reset(_vel_z_lp_filter.getState() + local_position.delta_vz);
			_vel_z_notch_filter.reset();
			_vz_reset_counter = local_position.vz_reset_counter;
		}

		if (local_position.heading_reset_counter != _heading_reset_counter) {
			_state.resets.heading = true;
			_state.resets.delta_heading = local_position.delta_heading;
			_heading_reset_counter = local_position.heading_reset_counter;
		}
	}

	// --- position ----------------------------------------------------------
	const Vector2f position_xy(local_position.x, local_position.y);

	if (local_position.xy_valid && position_xy.isAllFinite()) {
		_state.position.xy() = position_xy;
		_state.position_valid_xy = true;

	} else {
		_state.position(0) = _state.position(1) = NAN;
		_state.position_valid_xy = false;
	}

	if (PX4_ISFINITE(local_position.z) && local_position.z_valid) {
		_state.position(2) = local_position.z;
		_state.position_valid_z = true;

	} else {
		_state.position(2) = NAN;
		_state.position_valid_z = false;
	}

	// --- velocity and its derivative ---------------------------------------
	const Vector2f velocity_xy(local_position.vx, local_position.vy);

	if (local_position.v_xy_valid && velocity_xy.isAllFinite()) {
		const Vector2f vel_xy_prev = _vel_xy_lp_filter.getState();

		// vel xy notch filter, then low pass filter
		_state.velocity.xy() = _vel_xy_lp_filter.update(_vel_xy_notch_filter.apply(velocity_xy));

		// vel xy derivative low pass filter
		_state.acceleration.xy() = _vel_deriv_xy_lp_filter.update((_vel_xy_lp_filter.getState() - vel_xy_prev) / dt_s);
		_state.velocity_valid_xy = true;

	} else {
		_state.velocity(0) = _state.velocity(1) = NAN;
		_state.acceleration(0) = _state.acceleration(1) = NAN;
		_state.velocity_valid_xy = false;

		// reset filters to prevent acceleration spikes when regaining velocity
		_vel_xy_lp_filter.reset({});
		_vel_xy_notch_filter.reset();
		_vel_deriv_xy_lp_filter.reset({});
	}

	if (PX4_ISFINITE(local_position.vz) && local_position.v_z_valid) {
		const float vel_z_prev = _vel_z_lp_filter.getState();

		_state.velocity(2) = _vel_z_lp_filter.update(_vel_z_notch_filter.apply(local_position.vz));
		_state.acceleration(2) = _vel_deriv_z_lp_filter.update((_vel_z_lp_filter.getState() - vel_z_prev) / dt_s);
		_state.velocity_valid_z = true;

	} else {
		_state.velocity(2) = NAN;
		_state.acceleration(2) = NAN;
		_state.velocity_valid_z = false;

		_vel_z_lp_filter.reset({});
		_vel_z_notch_filter.reset();
		_vel_deriv_z_lp_filter.reset({});
	}

	_state.heading = local_position.heading;
	_state.unaided_heading = local_position.unaided_heading;
}

void VehicleStateProvider::updateAttitude(const vehicle_attitude_s &attitude)
{
	const float dt_s = (_last_attitude_sample == 0)
			   ? kDtAttitudeMax
			   : math::constrain((attitude.timestamp_sample - _last_attitude_sample) * 1e-6f,
					     kDtAttitudeMin, kDtAttitudeMax);
	_last_attitude_sample = attitude.timestamp_sample;

	_state.freshness.dt_attitude = dt_s;
	_state.freshness.attitude_new = true;
	_state.q = Quatf(attitude.q);

	// First sample only latches the counter; a fresh subscription is not a reset. Without
	// this the first vehicle_attitude after start or reset() fabricates a quaternion
	// reset with whatever delta_q_reset the estimator last published.
	if (!_quat_reset_counter_initialised) {
		_quat_reset_counter = attitude.quat_reset_counter;
		_quat_reset_counter_initialised = true;

	} else if (_quat_reset_counter != attitude.quat_reset_counter) {
		_state.resets.quat = true;
		_state.resets.delta_q = Quatf(attitude.delta_q_reset);
		_quat_reset_counter = attitude.quat_reset_counter;
	}
}

void VehicleStateProvider::updateAngularVelocity(const vehicle_angular_velocity_s &angular_velocity)
{
	const float dt_s = (_last_cycle_sample == 0)
			   ? kDtCycleMax
			   : math::constrain((angular_velocity.timestamp_sample - _last_cycle_sample) * 1e-6f,
					     kDtCycleMin, kDtCycleMax);
	_last_cycle_sample = angular_velocity.timestamp_sample;

	_state.timestamp_sample = angular_velocity.timestamp_sample;
	_state.freshness.dt = dt_s;
	_state.angular_velocity = Vector3f(angular_velocity.xyz);
	_state.angular_accel = Vector3f(angular_velocity.xyz_derivative);
}

void VehicleStateProvider::updateLandDetected(const vehicle_land_detected_s &land_detected)
{
	_state.landed = land_detected.landed;
	_state.maybe_landed = land_detected.maybe_landed;
	_state.ground_contact = land_detected.ground_contact;
	_state.freefall = land_detected.freefall;
}

void VehicleStateProvider::setArmed(bool armed, bool spooled_up)
{
	_state.armed = armed;
	_state.spooled_up = spooled_up;
}

void VehicleStateProvider::setHoverThrustEstimate(float hover_thrust)
{
	if (PX4_ISFINITE(hover_thrust)) {
		_hover_thrust_estimate = math::constrain(hover_thrust, 0.05f, 0.9f);
		_state.hover_thrust = _hover_thrust_estimate;
		_state.hover_thrust_valid = true;

	} else {
		_hover_thrust_estimate = NAN;
		_state.hover_thrust = _param_mpc_thr_hover.get();
		_state.hover_thrust_valid = false;
	}
}

const mc_ctrl::ControllerState &VehicleStateProvider::getState()
{
	if (!_state.hover_thrust_valid) {
		_state.hover_thrust = _param_mpc_thr_hover.get();
	}

	return _state;
}

void VehicleStateProvider::endCycle()
{
	_state.resets.clear();
	_state.freshness.position_new = false;
	_state.freshness.attitude_new = false;
}
