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
 * @file OuterLoop.hpp
 * @brief Trajectory-level work item on the nav_and_controllers work queue.
 *
 * Reproduces stock PX4's work-queue separation. In stock, mc_pos_control runs on
 * `nav_and_controllers` driven by vehicle_local_position and publishes
 * vehicle_attitude_setpoint; only mc_rate_control sits on the high-priority
 * `rate_ctrl` queue. This item restores that split:
 *
 *   OuterLoop  (nav_and_controllers, vehicle_local_position, ~100 Hz)
 *       position state path + trajectory front end + controller outer stage
 *       └─ publishes vehicle_attitude_setpoint ──┐
 *                                               │  (uORB — no shared state,
 *                                               │   so no locking on the RT path)
 *   MulticopterController (rate_ctrl, gyro rate) ◄┘
 *       attitude/rate state path + inner front end + controller inner stage
 *
 * Only active when the selected controller reports hasOuterStage(). A monolithic
 * full-stack law runs entirely on the rate_ctrl item instead, and this item then
 * does nothing.
 *
 * It owns its OWN VehicleStateProvider, CommandFrontEnd and parameter_update
 * subscription. Sharing any of those with the rate_ctrl item would be a data race
 * across work queues - including via the ModuleParams cascade, which is why this is
 * not a ModuleParams child of the module. The only object shared with the inner item
 * is the controller, whose two stages are required to touch disjoint state (see
 * MulticopterControllerBase::hasOuterStage).
 */

#pragma once

#include <CommandFrontEnd.hpp>
#include <ControllerIO.hpp>
#include <MulticopterControllerBase.hpp>
#include <VehicleStateProvider.hpp>

#include <lib/perf/perf_counter.h>
#include <px4_platform_common/atomic.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_work_queue/WorkItem.hpp>

#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionCallback.hpp>
#include <uORB/SubscriptionInterval.hpp>

#include <uORB/topics/hover_thrust_estimate.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/takeoff_status.h>
#include <uORB/topics/trajectory_setpoint.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_attitude_setpoint.h>
#include <uORB/topics/vehicle_constraints.h>
#include <uORB/topics/vehicle_control_mode.h>
#include <uORB/topics/vehicle_land_detected.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/vehicle_local_position_setpoint.h>
#include <uORB/topics/vehicle_status.h>

class OuterLoop : public ModuleParams, public px4::WorkItem
{
public:
	/**
	 * Deliberately NOT a ModuleParams child of the module: the cascade would refresh
	 * this item's parameters, filters and takeoff state from the rate_ctrl queue while
	 * this item is running on nav_and_controllers. It owns its own parameter_update
	 * subscription instead and refreshes from its own Run().
	 *
	 * @param active_controller shared with the rate_ctrl item; read, never written here
	 */
	explicit OuterLoop(px4::atomic<MulticopterControllerBase *> *active_controller);
	~OuterLoop() override;

	bool init();
	void stop();

	/// Takeoff state, for the inner item's status reporting.
	TakeoffState takeoffState() { return _front_end.takeoffState(); }

	/**
	 * True while Run() is between loading _active_controller and its last use of that
	 * pointer. The inner item must observe this false before freeing a retired
	 * controller, or it frees an object this item is still calling into.
	 */
	bool inUse() const { return _in_use.load(); }

	void printStatus();

private:
	void Run() override;

	px4::atomic<MulticopterControllerBase *> *_active_controller;
	px4::atomic_bool _in_use{false};

	VehicleStateProvider _state_provider{this};
	CommandFrontEnd _front_end{this};

	uORB::SubscriptionCallbackWorkItem _local_position_sub{this, ORB_ID(vehicle_local_position)};
	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _vehicle_attitude_sub{ORB_ID(vehicle_attitude)};
	uORB::Subscription _vehicle_control_mode_sub{ORB_ID(vehicle_control_mode)};
	uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
	uORB::Subscription _vehicle_land_detected_sub{ORB_ID(vehicle_land_detected)};
	uORB::Subscription _trajectory_setpoint_sub{ORB_ID(trajectory_setpoint)};
	uORB::Subscription _vehicle_constraints_sub{ORB_ID(vehicle_constraints)};
	uORB::Subscription _hover_thrust_estimate_sub{ORB_ID(hover_thrust_estimate)};

	uORB::Publication<vehicle_attitude_setpoint_s> _attitude_setpoint_pub{ORB_ID(vehicle_attitude_setpoint)};
	uORB::Publication<vehicle_local_position_setpoint_s> _local_position_setpoint_pub{ORB_ID(vehicle_local_position_setpoint)};
	uORB::Publication<takeoff_status_s> _takeoff_status_pub{ORB_ID(takeoff_status)};

	vehicle_control_mode_s _vehicle_control_mode{};
	vehicle_status_s _vehicle_status{};
	uint8_t _takeoff_state{0};
	uint32_t _update_count{0};
	bool _was_armed{false};
	mc_ctrl::ControlLevel _last_level{mc_ctrl::ControlLevel::None};
	MulticopterControllerBase *_last_controller{nullptr};

	perf_counter_t _loop_perf;

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::COM_SPOOLUP_TIME>) _param_com_spoolup_time
	)
};
