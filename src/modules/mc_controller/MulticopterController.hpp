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
 * @file MulticopterController.hpp
 * @brief Pluggable multicopter controller framework module.
 *
 * Runs on the rate_ctrl work queue driven by vehicle_angular_velocity, the same
 * cadence as mc_rate_control and control_allocator. Wires:
 *
 *   VehicleStateProvider -> CommandFrontEnd -> MulticopterControllerBase
 *                                                  -> output stage -> uORB
 *
 * The output stage is the safety boundary: it validates the controller's output,
 * latches a fallback to the reference cascade on any violation, and is the only
 * place that publishes.
 */

#pragma once

#include <CascadedPidController.hpp>
#include <CommandFrontEnd.hpp>
#include <ControllerRegistry.hpp>
#include <MulticopterControllerBase.hpp>
#include <VehicleStateProvider.hpp>

#include "OuterLoop.hpp"

#include <lib/perf/perf_counter.h>
#include <px4_platform_common/atomic.h>
#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_work_queue/WorkItem.hpp>

#include <uORB/Publication.hpp>
#include <uORB/PublicationMulti.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionCallback.hpp>
#include <uORB/SubscriptionInterval.hpp>

#include <uORB/topics/actuator_controls_status.h>
#include <uORB/topics/autotune_attitude_control_status.h>
#include <uORB/topics/battery_status.h>
#include <uORB/topics/control_allocator_status.h>
#include <uORB/topics/hover_thrust_estimate.h>
#include <uORB/topics/manual_control_setpoint.h>
#include <uORB/topics/mc_controller_status.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/rate_ctrl_status.h>
#include <uORB/topics/takeoff_status.h>
#include <uORB/topics/trajectory_setpoint.h>
#include <uORB/topics/vehicle_angular_velocity.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_attitude_setpoint.h>
#include <uORB/topics/vehicle_constraints.h>
#include <uORB/topics/vehicle_control_mode.h>
#include <uORB/topics/vehicle_land_detected.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/vehicle_local_position_setpoint.h>
#include <uORB/topics/vehicle_rates_setpoint.h>
#include <uORB/topics/vehicle_status.h>
#include <uORB/topics/vehicle_thrust_setpoint.h>
#include <uORB/topics/vehicle_torque_setpoint.h>

using namespace time_literals;

class MulticopterController : public ModuleBase, public ModuleParams, public px4::WorkItem
{
public:
	static Descriptor desc;

	MulticopterController();
	~MulticopterController() override;

	static int task_spawn(int argc, char *argv[]);
	static int custom_command(int argc, char *argv[]);
	static int print_usage(const char *reason = nullptr);
	int print_status() override;

	bool init();

private:
	void Run() override;

	void handleParameterUpdate();
	void pollInputs();
	bool instantiateController(int32_t alg);
	void reclaimRetiredController();
	void publishOutput(const mc_ctrl::ControllerOutput &output, uint64_t timestamp_sample, float dt);
	void publishIntermediateTopics(const CommandFrontEnd::Publications &publications);
	void publishStatus(float dt);
	void updateActuatorControlsStatus(const vehicle_torque_setpoint_s &torque_setpoint, float dt);
	void latchFallback(uint8_t reason, const char *why);

	// --- pipeline ----------------------------------------------------------
	VehicleStateProvider _state_provider{this};
	CommandFrontEnd _front_end{this};

	/// Always allocated so the fallback is instantaneous and allocation-free.
	CascadedPidController *_reference{nullptr};
	MulticopterControllerBase *_controller{nullptr};

	/**
	 * The controller currently in effect, shared with the outer-loop work item.
	 * Atomic because two work queues read it; it is only ever written from this item.
	 */
	px4::atomic<MulticopterControllerBase *> _active_controller{nullptr};

	/// Trajectory-level work item on nav_and_controllers — stock's queue separation.
	/// Not a ModuleParams child: it refreshes its own parameters on its own queue.
	OuterLoop _outer_loop{&_active_controller};

	/**
	 * Retired controller awaiting deletion.
	 *
	 * Freeing it inside instantiateController() would be a use-after-free: the outer
	 * item may already have loaded the pointer and be inside updateOuter(). We publish
	 * the replacement into _active_controller first and free the old one from Run(),
	 * once the outer item reports itself idle - at which point its next cycle is
	 * guaranteed to load the new pointer.
	 */
	MulticopterControllerBase *_controller_to_delete{nullptr};

	bool _fallback_latched{false};
	bool _was_armed{false};
	/// Set when the outer item stopped delivering and we took the trajectory stage back
	/// inline. Sticky for the armed period, like the fallback latch itself.
	bool _outer_stage_disabled{false};
	uint8_t _fallback_reason{mc_controller_status_s::FALLBACK_NONE};

	/// MC_CTRL_ALG value the live _controller was built from. Distinct from
	/// _pending_alg: conflating the two silently swallowed every deferred switch.
	int32_t _active_alg{-1};
	/// Requested MC_CTRL_ALG that could not be applied yet, or -1 for none.
	int32_t _pending_alg{-1};

	uint32_t _update_count{0};
	uint32_t _invalid_output_count{0};
	hrt_abstime _last_valid_output{0};
	mc_ctrl::ControlLevel _last_level{mc_ctrl::ControlLevel::None};

	// --- subscriptions -----------------------------------------------------
	uORB::SubscriptionCallbackWorkItem _vehicle_angular_velocity_sub{this, ORB_ID(vehicle_angular_velocity)};
	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _vehicle_attitude_sub{ORB_ID(vehicle_attitude)};
	uORB::Subscription _vehicle_local_position_sub{ORB_ID(vehicle_local_position)};
	uORB::Subscription _vehicle_control_mode_sub{ORB_ID(vehicle_control_mode)};
	uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
	uORB::Subscription _vehicle_land_detected_sub{ORB_ID(vehicle_land_detected)};
	uORB::Subscription _trajectory_setpoint_sub{ORB_ID(trajectory_setpoint)};
	uORB::Subscription _vehicle_constraints_sub{ORB_ID(vehicle_constraints)};
	uORB::Subscription _manual_control_setpoint_sub{ORB_ID(manual_control_setpoint)};
	uORB::Subscription _vehicle_attitude_setpoint_sub{ORB_ID(vehicle_attitude_setpoint)};
	uORB::Subscription _vehicle_rates_setpoint_sub{ORB_ID(vehicle_rates_setpoint)};
	uORB::Subscription _hover_thrust_estimate_sub{ORB_ID(hover_thrust_estimate)};
	uORB::Subscription _control_allocator_status_sub{ORB_ID(control_allocator_status)};
	uORB::Subscription _battery_status_sub{ORB_ID(battery_status)};
	uORB::Subscription _autotune_status_sub{ORB_ID(autotune_attitude_control_status)};

	// --- publications ------------------------------------------------------
	uORB::Publication<vehicle_torque_setpoint_s> _vehicle_torque_setpoint_pub{ORB_ID(vehicle_torque_setpoint)};
	uORB::Publication<vehicle_thrust_setpoint_s> _vehicle_thrust_setpoint_pub{ORB_ID(vehicle_thrust_setpoint)};
	uORB::Publication<vehicle_attitude_setpoint_s> _vehicle_attitude_setpoint_pub{ORB_ID(vehicle_attitude_setpoint)};
	uORB::Publication<vehicle_rates_setpoint_s> _vehicle_rates_setpoint_pub{ORB_ID(vehicle_rates_setpoint)};
	uORB::Publication<vehicle_local_position_setpoint_s> _local_position_setpoint_pub{ORB_ID(vehicle_local_position_setpoint)};
	uORB::Publication<takeoff_status_s> _takeoff_status_pub{ORB_ID(takeoff_status)};
	uORB::Publication<actuator_controls_status_s> _actuator_controls_status_pub{ORB_ID(actuator_controls_status_0)};
	uORB::Publication<mc_controller_status_s> _mc_controller_status_pub{ORB_ID(mc_controller_status)};
	uORB::PublicationMulti<rate_ctrl_status_s> _rate_ctrl_status_pub{ORB_ID(rate_ctrl_status)};

	// --- cached inputs -----------------------------------------------------
	vehicle_control_mode_s _vehicle_control_mode{};
	vehicle_status_s _vehicle_status{};
	float _battery_status_scale{0.f};
	uint8_t _takeoff_state{0};
	hrt_abstime _last_status_publish{0};

	// Last values actually published, mirrored into mc_controller_status. Recorded
	// after battery scaling so the log shows what the mixer saw, not what the control
	// law asked for.
	float _published_torque[3] {};
	float _published_thrust[3] {};

	// energy integration for actuator_controls_status_0
	float _energy_integration_time{0.f};
	float _control_energy[3] {};

	perf_counter_t _loop_perf;
	perf_counter_t _position_stage_perf;

	DEFINE_PARAMETERS(
		(ParamInt<px4::params::MC_CTRL_ALG>)      _param_mc_ctrl_alg,
		(ParamInt<px4::params::MC_CTRL_WD_MS>)    _param_mc_ctrl_wd_ms,
		(ParamBool<px4::params::MC_BAT_SCALE_EN>) _param_mc_bat_scale_en,
		(ParamFloat<px4::params::COM_SPOOLUP_TIME>) _param_com_spoolup_time
	)
};
