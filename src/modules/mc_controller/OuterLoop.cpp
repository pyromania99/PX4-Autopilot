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

#include "OuterLoop.hpp"

#include <drivers/drv_hrt.h>

using namespace matrix;
using namespace time_literals;

OuterLoop::OuterLoop(px4::atomic<MulticopterControllerBase *> *active_controller) :
	ModuleParams(nullptr),   // see the note on the constructor declaration
	WorkItem(MODULE_NAME "_outer", px4::wq_configurations::nav_and_controllers),
	_active_controller(active_controller),
	_loop_perf(perf_alloc(PC_ELAPSED, MODULE_NAME "_outer: cycle"))
{
	// The outer instance owns the trajectory path; the inner instance is told to
	// skip it and consume our published attitude setpoint instead.
	_front_end.setOuterStageActive(false);
}

OuterLoop::~OuterLoop()
{
	stop();
	perf_free(_loop_perf);
}

bool OuterLoop::init()
{
	if (!_local_position_sub.registerCallback()) {
		PX4_ERR("outer loop callback registration failed");
		return false;
	}

	return true;
}

void OuterLoop::stop()
{
	_local_position_sub.unregisterCallback();
}

void OuterLoop::Run()
{
	perf_begin(_loop_perf);

	vehicle_local_position_s local_position;

	if (!_local_position_sub.update(&local_position)) {
		perf_end(_loop_perf);
		return;
	}

	// Our own parameter refresh, on our own queue. The module's ModuleParams cascade
	// deliberately does not reach us.
	if (_parameter_update_sub.updated()) {
		parameter_update_s update;
		_parameter_update_sub.copy(&update);
		ModuleParams::updateParams();
	}

	// Publishing the flag before loading the pointer is what makes the inner item's
	// retire-then-free handshake sound: if it ever observes inUse() == false, we are
	// guaranteed to load _active_controller afresh on our next cycle.
	_in_use.store(true);
	MulticopterControllerBase *controller = _active_controller->load();

	if ((controller == nullptr) || !controller->hasOuterStage()) {
		// Monolithic controller: the rate_ctrl item runs the whole law, so there is
		// nothing for this item to do.
		_last_controller = controller;
		_in_use.store(false);
		perf_end(_loop_perf);
		return;
	}

	// A controller swap gets a clean outer integrator. Deliberately NOT _front_end.reset()
	// as well: that would rewind the takeoff state machine, and a rewind in flight puts
	// the command back to the on-ground override, which cuts thrust.
	if (controller != _last_controller) {
		controller->resetOuter();
		_last_controller = controller;
		_last_level = mc_ctrl::ControlLevel::None;
	}

	controller->updateOuterParams();

	_state_provider.updateLocalPosition(local_position);
	// This item is position-driven, so it owns the cycle timestamp. Without it the
	// takeoff state machine sees now == 0, never leaves rampup, and the climb-rate
	// limit never opens - the vehicle sits on the ground at zero thrust.
	_state_provider.setTimestampSample(local_position.timestamp_sample);

	// The attitude is needed for the yaw reference and for the front end's
	// mode-exit resets; it is read here but the quaternion-reset bookkeeping that
	// matters to the inner loop stays on the inner instance.
	vehicle_attitude_s attitude;

	if (_vehicle_attitude_sub.update(&attitude)) {
		_state_provider.updateAttitude(attitude);
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

	// Clear the outer stage's own state on the disarm edge, mirroring the inner item.
	if (_was_armed && !armed) {
		_front_end.reset();
		_state_provider.reset();
		controller->resetOuter();
	}

	_was_armed = armed;

	const mc_ctrl::ControllerState &state = _state_provider.getState();
	const uint64_t now = hrt_absolute_time();

	CommandFrontEnd::Publications publications;
	const mc_ctrl::ControllerCommand &command = _front_end.update(state, now, publications);

	// The inner item resets the controller on every level change; the outer stage needs
	// the same treatment, and only this queue may reset it.
	if (command.level != _last_level) {
		controller->resetOuter();
		_last_level = command.level;
	}

	// takeoff_status feeds land_detector, and must keep flowing regardless of level.
	const uint8_t takeoff_state = static_cast<uint8_t>(_front_end.takeoffState());

	if (takeoff_state != _takeoff_state) {
		_takeoff_state = takeoff_state;
		takeoff_status_s status{};
		status.takeoff_state = takeoff_state;
		status.timestamp = hrt_absolute_time();
		_takeoff_status_pub.publish(status);
	}

	if (command.level != mc_ctrl::ControlLevel::Trajectory) {
		// Not a position mode: the inner item owns the whole command.
		_state_provider.endCycle();
		_in_use.store(false);
		perf_end(_loop_perf);
		return;
	}

	vehicle_attitude_setpoint_s attitude_setpoint{};

	if (controller->updateOuter(state, command, state.freshness.dt_position, attitude_setpoint)) {
		attitude_setpoint.timestamp = hrt_absolute_time();
		_attitude_setpoint_pub.publish(attitude_setpoint);
		_update_count++;

		// Consumed by the flight tasks for smooth setpoint resets. Controllers with
		// no internal position setpoint leave the hook unimplemented.
		vehicle_local_position_setpoint_s local_setpoint{};
		controller->fillLocalPositionSetpoint(local_setpoint);
		local_setpoint.timestamp = attitude_setpoint.timestamp;
		_local_position_setpoint_pub.publish(local_setpoint);
	}

	// If updateOuter() failed we deliberately publish nothing: publishing a zeroed
	// setpoint would command a level attitude at zero thrust, which is worse than
	// holding the last good one briefly. The inner item notices the gap through
	// CommandFrontEnd::outerStageStale() and latches the reference fallback - the NaN
	// guard and the output watchdog cannot see this, because the inner controller keeps
	// producing a valid output from the held setpoint.

	_state_provider.endCycle();
	_in_use.store(false);
	perf_end(_loop_perf);
}

void OuterLoop::printStatus()
{
	MulticopterControllerBase *controller = _active_controller->load();
	const bool active = (controller != nullptr) && controller->hasOuterStage();

	PX4_INFO("outer loop       : %s (nav_and_controllers)", active ? "ACTIVE" : "idle - monolithic controller");
	PX4_INFO("outer updates    : %lu", (unsigned long)_update_count);
	perf_print_counter(_loop_perf);
}
