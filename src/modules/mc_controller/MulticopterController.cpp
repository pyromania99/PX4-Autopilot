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

#include "MulticopterController.hpp"

#include <drivers/drv_hrt.h>
#include <mathlib/math/Limits.hpp>
#include <systemlib/mavlink_log.h>

using namespace matrix;
using namespace time_literals;

ModuleBase::Descriptor MulticopterController::desc{task_spawn, custom_command, print_usage};

MulticopterController::MulticopterController() :
	ModuleParams(nullptr),
	WorkItem(MODULE_NAME, px4::wq_configurations::rate_ctrl),
	_loop_perf(perf_alloc(PC_ELAPSED, MODULE_NAME": cycle")),
	_position_stage_perf(perf_alloc(PC_ELAPSED, MODULE_NAME": position stage"))
{
	_reference = new CascadedPdController(this);
	_controller = _reference;
	_rate_ctrl_status_pub.advertise();
}

MulticopterController::~MulticopterController()
{
	// Take the outer item off its queue and out of the shared pointer before freeing
	// anything it might still be calling into.
	_active_controller.store(nullptr);
	_outer_loop.stop();

	delete _controller_to_delete;

	// _controller aliases _reference when MC_CTRL_ALG selects the reference cascade.
	if (_controller != _reference) {
		delete _controller;
	}

	delete _reference;

	perf_free(_loop_perf);
	perf_free(_position_stage_perf);
}

bool MulticopterController::init()
{
	if (_reference == nullptr) {
		PX4_ERR("reference controller allocation failed");
		return false;
	}

	if (!instantiateController(_param_mc_ctrl_alg.get())) {
		PX4_ERR("controller instantiation failed");
		return false;
	}

	if (!_outer_loop.init()) {
		return false;
	}

	// Registered last: this is what starts scheduling Run(), and everything Run()
	// touches has to exist by then.
	if (!_vehicle_angular_velocity_sub.registerCallback()) {
		PX4_ERR("callback registration failed");
		return false;
	}

	return true;
}

void MulticopterController::reclaimRetiredController()
{
	// Safe once the outer item reports itself idle: it sets that flag before loading
	// _active_controller and clears it after its last use, and we re-pointed
	// _active_controller before retiring, so its next cycle cannot reach this object.
	if ((_controller_to_delete != nullptr) && !_outer_loop.inUse()) {
		delete _controller_to_delete;
		_controller_to_delete = nullptr;
	}
}

bool MulticopterController::instantiateController(int32_t alg)
{
	// One retirement in flight at a time; that keeps the handshake below a simple
	// two-state one. The caller retries on the next cycle.
	reclaimRetiredController();

	if (_controller_to_delete != nullptr) {
		return false;
	}

	MulticopterControllerBase *previous = _controller;
	MulticopterControllerBase *next = mc_ctrl::createController(alg, this, _reference);

	if (next == nullptr) {
		return false;
	}

	_controller = next;
	_controller->reset();
	_reference->reset();
	_fallback_latched = false;
	_fallback_reason = mc_controller_status_s::FALLBACK_NONE;

	// Publish to the outer item BEFORE retiring the old instance, so the outer item
	// can never load a pointer we are about to free.
	_active_controller.store(_controller);
	_front_end.setOuterStageActive(_controller->hasOuterStage());

	if ((previous != nullptr) && (previous != _reference) && (previous != next)) {
		// Freed from Run() once the outer item goes idle - never here, where it may
		// still be inside previous->updateOuter().
		_controller_to_delete = previous;
	}

	_active_alg = alg;
	_pending_alg = -1;

	if (_param_mc_ctrl_gt.get() != 0) {
		PX4_WARN("MC_CTRL_GT=1: flying on SIMULATOR GROUND TRUTH, not the estimator");
		mavlink_log_critical(nullptr, "MC_CTRL_GT: controller on ground truth\t");
	}

	PX4_INFO("controller: %s (MC_CTRL_ALG=%d), levels=0x%02x, queues=%s",
		 _controller->name(), (int)alg, _controller->supportedLevels(),
		 _controller->hasOuterStage() ? "split (stock separation)" : "single (gyro rate)");

	// Surface an unsupported-level configuration on the ground rather than at the
	// first mode switch in the air.
	if (_controller->supportedLevels() != mc_ctrl::kAllLevels) {
		PX4_WARN("%s does not support all control levels (0x%02x); the reference cascade will take over for the rest",
			 _controller->name(), _controller->supportedLevels());
	}

	return true;
}

void MulticopterController::handleParameterUpdate()
{
	if (_parameter_update_sub.updated()) {
		parameter_update_s update;
		_parameter_update_sub.copy(&update);
		updateParams();

		const int32_t alg = _param_mc_ctrl_alg.get();

		// _active_alg is what is actually running; comparing against it (rather than
		// against the pending request, as this once did) is what makes a deferred
		// switch survive to be applied.
		if (alg != _active_alg) {
			if (_pending_alg != alg) {
				PX4_WARN("MC_CTRL_ALG change to %d deferred until disarm", (int)alg);
			}

			_pending_alg = alg;
		}
	}

	// Apply a pending switch as soon as we are disarmed. Never allocate while armed.
	// instantiateController() clears _pending_alg on success and leaves it set when it
	// cannot run yet, so this simply retries next cycle.
	if ((_pending_alg != -1) && !_vehicle_control_mode.flag_armed) {
		instantiateController(_pending_alg);
	}
}

void MulticopterController::pollInputs()
{
	// MC_CTRL_GT swaps the state source for simulator ground truth. Read once per cycle
	// into a local rather than branching twice below, and note the ground-truth topics are
	// the SAME message types - the substitution ends at the subscription.
	const bool use_groundtruth = (_param_mc_ctrl_gt.get() != 0);

	vehicle_attitude_s attitude;

	if (use_groundtruth ? _vehicle_attitude_gt_sub.update(&attitude)
	    : _vehicle_attitude_sub.update(&attitude)) {
		_state_provider.updateAttitude(attitude);
	}

	vehicle_local_position_s local_position;

	if (use_groundtruth ? _vehicle_local_position_gt_sub.update(&local_position)
	    : _vehicle_local_position_sub.update(&local_position)) {
		// Kept running even while the outer item owns the position stage and nothing
		// here reads the result. Skipping it would save the velocity filter chain on the
		// rate_ctrl queue - a notch, two low passes and a derivative at ~100 Hz, so a
		// fraction of a percent of the budget - but it would leave the filters cold and
		// _state.position at the origin. The stale-outer-stage takeover below then runs
		// the position stage inline against zeroed state and a derivative that steps
		// from 0 to the true velocity in one sample. Paying for warm state on the RT
		// queue is much cheaper than an acceleration spike on the recovery path.
		_state_provider.updateLocalPosition(local_position);
	}

	vehicle_land_detected_s land_detected;

	if (_vehicle_land_detected_sub.update(&land_detected)) {
		_state_provider.updateLandDetected(land_detected);
	}

	if (_vehicle_control_mode_sub.update(&_vehicle_control_mode)) {
		_front_end.setControlMode(_vehicle_control_mode);
	}

	if (_vehicle_status_sub.update(&_vehicle_status)) {
		_front_end.setVehicleStatus(_vehicle_status);
	}

	const bool armed = (_vehicle_status.arming_state == vehicle_status_s::ARMING_STATE_ARMED);
	const bool spooled_up = armed
				&& (hrt_elapsed_time(&_vehicle_status.armed_time) > _param_com_spoolup_time.get() * 1_s);
	_state_provider.setArmed(armed, spooled_up);

	hover_thrust_estimate_s hover_thrust_estimate;

	if (_hover_thrust_estimate_sub.update(&hover_thrust_estimate)) {
		// Single entry point. The front end also needs this - stock feeds it to the
		// manual throttle curve separately - but it reads it off state.hover_thrust
		// rather than requiring a second call here, so the two can never diverge.
		_state_provider.setHoverThrustEstimate(hover_thrust_estimate.valid ? hover_thrust_estimate.hover_thrust : NAN);
	}

	trajectory_setpoint_s trajectory_setpoint;

	if (_trajectory_setpoint_sub.update(&trajectory_setpoint)) {
		_front_end.setTrajectorySetpoint(trajectory_setpoint);
	}

	vehicle_constraints_s constraints;

	if (_vehicle_constraints_sub.update(&constraints)) {
		_front_end.setVehicleConstraints(constraints);
	}

	manual_control_setpoint_s manual_control_setpoint;

	if (_manual_control_setpoint_sub.update(&manual_control_setpoint)) {
		_front_end.setManualControlSetpoint(manual_control_setpoint);
	}

	vehicle_attitude_setpoint_s attitude_setpoint;

	if (_vehicle_attitude_setpoint_sub.update(&attitude_setpoint)) {
		_front_end.setExternalAttitudeSetpoint(attitude_setpoint);
	}

	vehicle_rates_setpoint_s rates_setpoint;

	if (_vehicle_rates_setpoint_sub.update(&rates_setpoint)) {
		_front_end.setExternalRatesSetpoint(rates_setpoint);
	}

	// Allocator anti-windup feedback, offered to the controller as an optional hook.
	control_allocator_status_s allocator_status;

	if (_control_allocator_status_sub.update(&allocator_status)) {
		mc_ctrl::AllocatorFeedback feedback;
		feedback.torque_setpoint_achieved = allocator_status.torque_setpoint_achieved;
		feedback.unallocated_torque = Vector3f(allocator_status.unallocated_torque);

		if (!allocator_status.torque_setpoint_achieved) {
			for (size_t i = 0; i < 3; i++) {
				if (allocator_status.unallocated_torque[i] > FLT_EPSILON) {
					feedback.saturation_positive(i) = true;

				} else if (allocator_status.unallocated_torque[i] < -FLT_EPSILON) {
					feedback.saturation_negative(i) = true;
				}
			}
		}

		if (_controller != nullptr) {
			_controller->setAllocatorFeedback(feedback);
		}

		if (_reference != _controller) {
			_reference->setAllocatorFeedback(feedback);
		}
	}

	// Autotune injects an additive rate setpoint at the attitude stage.
	autotune_attitude_control_status_s autotune;

	if (_autotune_status_sub.copy(&autotune)) {
		const bool active = (autotune.state == autotune_attitude_control_status_s::STATE_ROLL
				     || autotune.state == autotune_attitude_control_status_s::STATE_PITCH
				     || autotune.state == autotune_attitude_control_status_s::STATE_YAW
				     || autotune.state == autotune_attitude_control_status_s::STATE_TEST)
				    && ((hrt_absolute_time() - autotune.timestamp) < 1_s);

		if (active) {
			_reference->setAutotuneRateSetpoint(Vector3f(autotune.rate_sp));

		} else {
			_reference->clearAutotuneRateSetpoint();
		}
	}
}

void MulticopterController::latchFallback(uint8_t reason, const char *why)
{
	if (_fallback_latched) {
		return;
	}

	_fallback_latched = true;
	_fallback_reason = reason;
	PX4_ERR("controller fallback: %s", why);
	mavlink_log_critical(nullptr, "MC controller fallback: %s\t", why);
}

void MulticopterController::Run()
{
	if (should_exit()) {
		_vehicle_angular_velocity_sub.unregisterCallback();
		_outer_loop.stop();
		exit_and_cleanup(desc);
		return;
	}

	perf_begin(_loop_perf);

	handleParameterUpdate();

	vehicle_angular_velocity_s angular_velocity;

	if (!_vehicle_angular_velocity_sub.update(&angular_velocity)) {
		perf_end(_loop_perf);
		return;
	}

	_state_provider.updateAngularVelocity(angular_velocity);
	pollInputs();

	// A controller retired by an earlier MC_CTRL_ALG switch is freed here, once the
	// outer item is idle - never at the point of the switch itself.
	reclaimRetiredController();

	const mc_ctrl::ControllerState &state = _state_provider.getState();
	const uint64_t now = hrt_absolute_time();
	const float dt = state.freshness.dt;

	// The fallback latch is sticky for the armed period: a NaN out of a research
	// controller means corrupted internal state, and retrying next cycle would
	// produce a limit cycle between garbage and fallback.
	//
	// Cleared on the disarm TRANSITION, not while disarmed: a level-triggered clear
	// re-latches every cycle against a persistently bad controller, which both
	// defeats the stickiness and spams the log at gyro rate.
	const bool armed_now = _vehicle_control_mode.flag_armed;

	if (_was_armed && !armed_now) {
		if (_fallback_latched) {
			_fallback_latched = false;
			_fallback_reason = mc_controller_status_s::FALLBACK_NONE;
			_controller->reset();
			PX4_INFO("fallback cleared on disarm");
		}

		if (_outer_stage_disabled) {
			// Hand the trajectory stage back to the outer item.
			_outer_stage_disabled = false;
			_active_controller.store(_controller);
			_front_end.setOuterStageActive(_controller->hasOuterStage());
		}

		// Re-arm on a clean slate. Carrying the previous flight's timestamp across the
		// disarm means the first armed cycle measures the whole disarmed interval and
		// latches the watchdog on a controller that has done nothing wrong.
		_last_valid_output = 0;
	}

	_was_armed = armed_now;

	// Watchdog: no valid output for MC_CTRL_WD_MS while armed. Evaluated up front, on
	// every cycle, rather than after a successful update - down there it could only
	// ever run on a cycle that had just succeeded, which made it unreachable.
	if (armed_now && (_last_valid_output != 0)
	    && (hrt_elapsed_time(&_last_valid_output) > (hrt_abstime)_param_mc_ctrl_wd_ms.get() * 1000)) {
		latchFallback(mc_controller_status_s::FALLBACK_STALE, "controller output watchdog");
	}

	CommandFrontEnd::Publications publications;
	const mc_ctrl::ControllerCommand &command = _front_end.update(state, now, publications);

	if (command.level != _last_level) {
		_controller->reset();
		_reference->reset();
		_last_level = command.level;
	}

	publishIntermediateTopics(publications);

	if (command.level == mc_ctrl::ControlLevel::None) {
		_state_provider.endCycle();
		perf_end(_loop_perf);
		return;
	}

	if (_controller == nullptr) {
		latchFallback(mc_controller_status_s::FALLBACK_UNINITIALISED, "no controller instance");
	}

	MulticopterControllerBase *active = _fallback_latched ? _reference : _controller;

	// The reference cascade is split; a fallback from a monolithic controller must
	// therefore switch the front end over, or the trajectory path would be run by
	// neither item.
	if (!_outer_stage_disabled && (_front_end.outerStageActive() != active->hasOuterStage())) {
		_front_end.setOuterStageActive(active->hasOuterStage());
		_active_controller.store(active);
	}

	// A level outside the controller's declared support latches the fallback.
	if (!_fallback_latched && !active->supportsLevel(command.level)) {
		latchFallback(mc_controller_status_s::FALLBACK_UNSUPPORTED_LEVEL, "unsupported control level");
		active = _reference;
	}

	// The outer item stopped delivering attitude setpoints. Nothing else can detect
	// this: `active` keeps tracking the held setpoint and reports a perfectly valid
	// output, so neither the NaN guard nor the watchdog above would ever fire, and the
	// vehicle would fly the last commanded attitude indefinitely.
	//
	// Falling back to the reference is not enough on its own - it is split too, so it
	// would consume the same stale setpoint. Take the outer item out of the loop and
	// run the trajectory stage inline here for the rest of the armed period.
	if (!_outer_stage_disabled && armed_now && _front_end.outerStageStale()) {
		latchFallback(mc_controller_status_s::FALLBACK_STALE, "outer loop stopped publishing");
		active = _reference;

		// Same handshake as retiring a controller: take the position stage over only
		// while the outer item is between cycles. Flipping it mid-cycle would leave
		// _position_control running on both queues at once - which is precisely the
		// race the whole outer/inner split exists to avoid. The usual cause of this
		// branch is updateOuter() returning false, where the outer item is cycling
		// normally and idle most of the time, so the takeover lands within a cycle or
		// two; retried every cycle until it does.
		if (!_outer_loop.inUse()) {
			_outer_stage_disabled = true;
			_active_controller.store(nullptr);
			_front_end.setOuterStageActive(false);
			active->reset();
		}
	}

	if (state.freshness.position_new) {
		perf_begin(_position_stage_perf);
	}

	mc_ctrl::ControllerOutput output;
	const bool ok = active->update(state, command, dt, output);

	if (state.freshness.position_new) {
		perf_end(_position_stage_perf);
	}

	const bool finite = mc_ctrl::outputIsFinite(output);

	if (!ok || !output.valid || !finite) {
		_invalid_output_count++;

		if (!_fallback_latched) {
			latchFallback(mc_controller_status_s::FALLBACK_INVALID_OUTPUT, "invalid controller output");
			// Re-run on the reference this cycle so the vehicle is never left without a command.
			output.reset();
			_reference->reset();

			if (!_reference->update(state, command, dt, output)
			    || !mc_ctrl::outputIsFinite(output)) {
				_state_provider.endCycle();
				perf_end(_loop_perf);
				return;
			}

			active = _reference;

		} else {
			_state_provider.endCycle();
			perf_end(_loop_perf);
			return;
		}
	}

	_last_valid_output = now;
	_update_count++;

	publishOutput(output, state.timestamp_sample, dt);
	publishStatus(dt);

	_state_provider.endCycle();
	perf_end(_loop_perf);
}

void MulticopterController::publishIntermediateTopics(const CommandFrontEnd::Publications &publications)
{
	if (publications.attitude_setpoint) {
		vehicle_attitude_setpoint_s sp = publications.attitude_setpoint_out;
		sp.timestamp = hrt_absolute_time();
		_vehicle_attitude_setpoint_pub.publish(sp);
	}

	if (publications.rates_setpoint) {
		vehicle_rates_setpoint_s sp = publications.rates_setpoint_out;
		sp.timestamp = hrt_absolute_time();
		_vehicle_rates_setpoint_pub.publish(sp);
	}

	// takeoff_status feeds land_detector. When the outer item is active it owns the
	// takeoff state machine and publishes this itself.
	if (_front_end.outerStageActive()) {
		return;
	}

	const uint8_t takeoff_state = static_cast<uint8_t>(_front_end.takeoffState());

	if (takeoff_state != _takeoff_state) {
		_takeoff_state = takeoff_state;
		takeoff_status_s status{};
		status.takeoff_state = takeoff_state;
		status.timestamp = hrt_absolute_time();
		_takeoff_status_pub.publish(status);
	}
}

void MulticopterController::publishOutput(const mc_ctrl::ControllerOutput &output, uint64_t timestamp_sample, float dt)
{
	vehicle_torque_setpoint_s torque_setpoint{};
	vehicle_thrust_setpoint_s thrust_setpoint{};

	// NaN was already rejected by outputIsFinite(); this mirrors the stock guard.
	for (int i = 0; i < 3; i++) {
		torque_setpoint.xyz[i] = PX4_ISFINITE(output.torque(i)) ? output.torque(i) : 0.f;
		thrust_setpoint.xyz[i] = PX4_ISFINITE(output.thrust(i)) ? output.thrust(i) : 0.f;
	}

	// Battery scaling (MulticopterRateControl.cpp:243-258).
	if (_param_mc_bat_scale_en.get()) {
		battery_status_s battery_status;

		if (_battery_status_sub.updated() && _battery_status_sub.copy(&battery_status)
		    && battery_status.connected && battery_status.scale > 0.f) {
			_battery_status_scale = battery_status.scale;
		}

		if (_battery_status_scale > 0.f) {
			for (int i = 0; i < 3; i++) {
				torque_setpoint.xyz[i] = math::constrain(torque_setpoint.xyz[i] * _battery_status_scale, -1.f, 1.f);
				thrust_setpoint.xyz[i] = math::constrain(thrust_setpoint.xyz[i] * _battery_status_scale, -1.f, 1.f);
			}
		}
	}

	const hrt_abstime now = hrt_absolute_time();

	thrust_setpoint.timestamp_sample = timestamp_sample;
	thrust_setpoint.timestamp = now;
	_vehicle_thrust_setpoint_pub.publish(thrust_setpoint);

	torque_setpoint.timestamp_sample = timestamp_sample;
	torque_setpoint.timestamp = now;
	_vehicle_torque_setpoint_pub.publish(torque_setpoint);

	// Mirror what actually went out, for mc_controller_status.
	memcpy(_published_torque, torque_setpoint.xyz, sizeof(_published_torque));
	memcpy(_published_thrust, thrust_setpoint.xyz, sizeof(_published_thrust));

	MulticopterControllerBase *active = _fallback_latched ? _reference : _controller;

	// Intermediate setpoints that stock publishes and other modules consume
	// (WeatherVane, mavlink ATTITUDE_TARGET, gimbal stream, QGC).
	//
	// Published only where WE generated them, never where we consumed them, or a
	// self-feedback path would form:
	//   Trajectory : we generate both the attitude and the rate setpoint
	//   Attitude   : we generate the rate setpoint; the attitude setpoint is either
	//                ours (manual sticks, already published by the front end) or
	//                external (offboard/VTOL) and must not be echoed
	//   BodyRate   : the rate setpoint is either ours (acro, published by the front
	//                end) or external, so nothing is published here
	if ((_last_level == mc_ctrl::ControlLevel::Trajectory) && !_front_end.outerStageActive()) {
		if (output.attitude_setpoint.isAllFinite()) {
			vehicle_attitude_setpoint_s attitude_setpoint{};
			output.attitude_setpoint.copyTo(attitude_setpoint.q_d);
			output.thrust.copyTo(attitude_setpoint.thrust_body);
			attitude_setpoint.timestamp = now;
			_vehicle_attitude_setpoint_pub.publish(attitude_setpoint);
		}
	}

	if ((_last_level == mc_ctrl::ControlLevel::Trajectory || _last_level == mc_ctrl::ControlLevel::Attitude)
	    && output.rate_setpoint.isAllFinite()) {
		vehicle_rates_setpoint_s rates_setpoint{};
		rates_setpoint.roll = output.rate_setpoint(0);
		rates_setpoint.pitch = output.rate_setpoint(1);
		rates_setpoint.yaw = output.rate_setpoint(2);
		output.thrust.copyTo(rates_setpoint.thrust_body);
		rates_setpoint.timestamp = now;
		_vehicle_rates_setpoint_pub.publish(rates_setpoint);
	}

	// Diagnostics consumed by mc_autotune and the logger.
	if (active == _reference || _controller == _reference) {
		rate_ctrl_status_s rate_status{};
		_reference->getRateControlStatus(rate_status);
		rate_status.timestamp = now;
		_rate_ctrl_status_pub.publish(rate_status);
	}

	// vehicle_local_position_setpoint belongs to whichever item ran the position
	// stage: the outer item when it is active, this one otherwise. Gated on
	// Trajectory for the same reason as the attitude setpoint above - stock only
	// publishes this while mc_pos_control runs, and mc_pos_control does not run
	// outside a position mode. Publishing a held setpoint through Stabilized/Acro
	// hands the flight tasks a stale reset origin and makes altitude tracking look
	// broken in the log even though the vehicle is flying correctly.
	//
	// Filled through the base-class hook rather than off _reference, so a third-party
	// controller that runs its own position stage on this queue still gets its setpoint
	// published. Gating this on `active == _reference` was invisible while the reference
	// was the only single-queue controller, but it silently starves the topic for any
	// other one - the exact starvation the paragraph above describes. A controller with no
	// internal position setpoint leaves the hook unimplemented and publishes the zeroed
	// struct, matching what OuterLoop already does on the split path.
	if ((_last_level == mc_ctrl::ControlLevel::Trajectory) && !_front_end.outerStageActive()) {
		vehicle_local_position_setpoint_s local_sp{};
		active->fillLocalPositionSetpoint(local_sp);
		local_sp.timestamp = now;
		_local_position_setpoint_pub.publish(local_sp);
	}

	updateActuatorControlsStatus(torque_setpoint, dt);
}

void MulticopterController::updateActuatorControlsStatus(const vehicle_torque_setpoint_s &torque_setpoint, float dt)
{
	for (int i = 0; i < 3; i++) {
		_control_energy[i] += torque_setpoint.xyz[i] * torque_setpoint.xyz[i] * dt;
	}

	_energy_integration_time += dt;

	if (_energy_integration_time > 500e-3f) {
		actuator_controls_status_s status{};

		for (int i = 0; i < 3; i++) {
			status.control_power[i] = _control_energy[i] / _energy_integration_time;
			_control_energy[i] = 0.f;
		}

		status.timestamp = hrt_absolute_time();
		_actuator_controls_status_pub.publish(status);
		_energy_integration_time = 0.f;
	}
}

void MulticopterController::publishStatus(float dt)
{
	const hrt_abstime now = hrt_absolute_time();

	if ((now - _last_status_publish) < 20_ms) {   // ~50 Hz
		return;
	}

	_last_status_publish = now;

	MulticopterControllerBase *active = _fallback_latched ? _reference : _controller;
	const mc_ctrl::ControllerState &state = _state_provider.getState();

	mc_controller_status_s status{};
	// The algorithm actually instantiated, not the parameter: while a switch is pending
	// those differ, and reporting the parameter would claim a controller that is not
	// running.
	status.algorithm = (uint8_t)_active_alg;
	status.control_level = (uint8_t)_last_level;
	status.fallback_active = _fallback_latched;
	status.fallback_reason = _fallback_reason;

	memcpy(status.torque_sp, _published_torque, sizeof(status.torque_sp));
	memcpy(status.thrust_sp, _published_thrust, sizeof(status.thrust_sp));

	status.dt = dt;
	status.dt_attitude = state.freshness.dt_attitude;
	status.dt_position = state.freshness.dt_position;
	status.update_count = _update_count;
	status.invalid_output_count = _invalid_output_count;

	active->fillStatus(status);

	status.timestamp = hrt_absolute_time();
	_mc_controller_status_pub.publish(status);
}

int MulticopterController::task_spawn(int argc, char *argv[])
{
	MulticopterController *instance = new MulticopterController();

	if (instance) {
		desc.object.store(instance);
		desc.task_id = task_id_is_work_queue;

		if (instance->init()) {
			return PX4_OK;
		}

	} else {
		PX4_ERR("alloc failed");
	}

	delete instance;
	desc.object.store(nullptr);
	desc.task_id = -1;

	return PX4_ERROR;
}

int MulticopterController::custom_command(int argc, char *argv[])
{
	return print_usage("unknown command");
}

int MulticopterController::print_status()
{
	MulticopterControllerBase *active = _fallback_latched ? _reference : _controller;

	PX4_INFO("MC_CTRL_ALG      : %d (%s)", (int)_active_alg, mc_ctrl::algorithmName(_active_alg));
	PX4_INFO("active controller: %s", active ? active->name() : "none");
	PX4_INFO("supported levels : 0x%02x", active ? active->supportedLevels() : 0);
	PX4_INFO("control level    : %s", mc_ctrl::levelName(_last_level));
	PX4_INFO("fallback         : %s (reason %d)", _fallback_latched ? "LATCHED" : "no", _fallback_reason);
	PX4_INFO("outer stage      : %s", _outer_stage_disabled ? "DISABLED (stopped publishing)" : "normal");
	PX4_INFO("updates          : %lu", (unsigned long)_update_count);
	PX4_INFO("invalid outputs  : %lu", (unsigned long)_invalid_output_count);

	if (_pending_alg != -1) {
		PX4_INFO("pending alg      : %d (%s, applies on disarm)", (int)_pending_alg,
			 mc_ctrl::algorithmName(_pending_alg));
	}

	_outer_loop.printStatus();

	if (active) {
		active->printStatus();
	}

	perf_print_counter(_loop_perf);
	perf_print_counter(_position_stage_perf);
	return 0;
}

int MulticopterController::print_usage(const char *reason)
{
	if (reason) {
		PX4_WARN("%s\n", reason);
	}

	PRINT_MODULE_DESCRIPTION(
		R"DESCR_STR(
### Description
Pluggable multicopter controller framework.

Takes the estimator state and the commanded setpoint (at whatever level the active
flight mode commands) and produces motor moments, using a control law selected at
runtime by MC_CTRL_ALG.

MC_CTRL_ALG=0 does not start this module; the stock mc_pos_control / mc_att_control /
mc_rate_control chain runs instead. Any non-zero value starts this module in place
of that chain.

)DESCR_STR");

	PRINT_MODULE_USAGE_NAME("mc_controller", "controller");
	PRINT_MODULE_USAGE_COMMAND("start");
	PRINT_MODULE_USAGE_DEFAULT_COMMANDS();

	return 0;
}

extern "C" __EXPORT int mc_controller_main(int argc, char *argv[])
{
	return ModuleBase::main(MulticopterController::desc, argc, argv);
}
