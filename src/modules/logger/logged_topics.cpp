/****************************************************************************
 *
 *   Copyright (c) 2019-2022 PX4 Development Team. All rights reserved.
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

#include "logged_topics.h"
#include "messages.h"

#include <parameters/param.h>
#include <px4_platform_common/log.h>
#include <px4_platform_common/px4_config.h>
#include <uORB/topics/uORBTopics.hpp>

#include <string.h>

using namespace px4::logger;

void LoggedTopics::add_default_topics()
{
	// TRIMMED for this fork, 2026-09-23. Upstream's default set grew past
	// MAX_TOPICS_NUM (255) and the logger then silently dropped whatever was added
	// last - which was the SITL *_groundtruth block, i.e. exactly the topics every
	// analysis in the PegasusSimulator tree scores on. It reported the loss only as
	// "Too many subscriptions, failed to add: vehicle_local_position_groundtruth 0"
	// in the boot log, and produced ulogs that looked complete.
	//
	// What is in here now: flight state, the control and allocation chain, the
	// experiment's own mc_controller_status, and EV pose. What moved OUT, and where
	// to turn it back on:
	//
	//   every estimator* topic  -> add_estimator_replay_topics()  SDLOG_PROFILE +2
	//   raw IMU / vibration     -> add_sensor_comparison_topics() SDLOG_PROFILE +64
	//
	// What was deleted outright: fixed-wing, VTOL, rover, gimbal, camera, ADSB,
	// follow-target, landing-gear, iridium, irlock, transponder, generator, heater,
	// internal-combustion, hygrometer, GNSS RF/spectrum blocks, fuel tank, dronecan,
	// px4io, cellular, VTX and the vision-target estimator. None of it exists on this
	// airframe.

	// --- commander / mode / arming ---------------------------------------------
	add_topic("action_request");
	add_topic("actuator_armed");
	add_topic("commander_state");
	add_topic("failsafe_flags");
	add_topic("home_position");
	add_topic("parameter_update");
	add_topic("vehicle_command");
	add_topic("vehicle_command_ack");
	add_topic("vehicle_control_mode");
	add_topic("vehicle_land_detected");
	add_topic("vehicle_status");
	add_topic("failure_detector_status", 100);
	add_optional_topic("takeoff_status", 1000);

	// --- pilot / offboard input -------------------------------------------------
	add_topic("input_rc", 500);
	add_topic("manual_control_setpoint", 200);
	add_topic("manual_control_switches");	// carries kill_switch_2, the rotor-failure hook
	add_topic("offboard_control_mode", 100);
	add_topic("goto_setpoint", 200);
	add_topic("position_setpoint_triplet", 200);
	add_topic("trajectory_setpoint", 200);

	// --- state ------------------------------------------------------------------
	add_topic("vehicle_acceleration", 50);
	add_topic("vehicle_angular_velocity", 20);
	add_topic("vehicle_attitude", 50);
	add_topic("vehicle_global_position", 200);
	add_topic("vehicle_gps_position", 100);
	add_topic("vehicle_local_position", 100);
	add_topic("sensor_combined");
	add_topic("sensor_selection");
	add_topic("wind", 1000);

	// External vision pose - kept deliberately: it is an input to the estimator and
	// the only pose source that does not come from the EKF itself.
	//
	// UNTHROTTLED on purpose. EKF2_EV_DELAY is calibrated by cross-correlating this
	// against the gyro (notes/HARDWARE_READINESS.md 2b), and the answer is of order
	// 10 ms, so a 30 ms log interval cannot resolve it. The bridge publishes at
	// ~119 Hz, which costs a fraction of what sensor_gyro_fifo did.
	add_optional_topic("vehicle_visual_odometry");

	// --- control chain ----------------------------------------------------------
	add_optional_topic("mc_controller_status", 20);
	add_topic("vehicle_attitude_setpoint", 50);
	add_topic("vehicle_rates_setpoint", 20);
	add_topic("vehicle_local_position_setpoint", 100);
	add_topic("hover_thrust_estimate", 100);
	add_optional_topic("actuator_controls_status_0", 300);
	add_optional_topic_multi("rate_ctrl_status", 200, 2);

	// --- allocation / actuators -------------------------------------------------
	add_topic("actuator_motors", 100);
	add_topic("actuator_servos", 100);
	add_topic_multi("vehicle_thrust_setpoint", 20, 2);
	add_topic_multi("vehicle_torque_setpoint", 20, 2);
	add_optional_topic_multi("actuator_outputs", 100, 3);
	add_optional_topic_multi("control_allocator_status", 200, 2);
	add_topic("esc_status", 100);

	// --- housekeeping -----------------------------------------------------------
	add_topic("cpuload");
	add_topic("logger_status", 200);
	add_topic_multi("battery_status", 200, 3);

	// SYS_HITL: default ground truth logging for simulation
	int32_t sys_hitl = 0;
	param_get(param_find("SYS_HITL"), &sys_hitl);

	if (sys_hitl >= 1) {
		add_topic("vehicle_angular_velocity_groundtruth", 10);
		add_topic("vehicle_attitude_groundtruth", 10);
		add_topic("vehicle_global_position_groundtruth", 100);
		add_topic("vehicle_local_position_groundtruth", 20);
	}

#ifdef CONFIG_ARCH_BOARD_PX4_SITL
	// Ground truth FIRST in this block. It is the whole point of a SITL log, and
	// being added last is what got it dropped.
	add_topic("vehicle_angular_velocity_groundtruth", 10);
	add_topic("vehicle_attitude_groundtruth", 10);
	add_topic("vehicle_global_position_groundtruth", 100);
	add_topic("vehicle_local_position_groundtruth", 20);

	add_topic("vehicle_angular_velocity", 10);	// re-add at the faster SITL rate
	add_topic("time_offset");
#endif /* CONFIG_ARCH_BOARD_PX4_SITL */
}

void LoggedTopics::add_high_rate_topics()
{
	// maximum rate to analyze fast maneuvers (e.g. for racing)
	add_topic("manual_control_setpoint");
	add_topic_multi("rate_ctrl_status", 20, 2);
	add_topic("sensor_combined");
	add_topic("vehicle_angular_velocity");
	add_topic("vehicle_attitude");
	add_topic("vehicle_attitude_setpoint");
	add_topic("vehicle_rates_setpoint");

	add_topic("esc_status", 5);
	add_topic("actuator_motors");
	add_topic("actuator_outputs_debug");
	add_topic("actuator_servos");
	add_topic_multi("vehicle_thrust_setpoint", 0, 2);
	add_topic_multi("vehicle_torque_setpoint", 0, 2);
}

void LoggedTopics::add_debug_topics()
{
	add_topic("debug_array");
	add_topic("debug_key_value");
	add_topic("debug_value");
	add_topic("debug_vect");
	add_topic_multi("satellite_info", 1000, 2);
	add_topic("mag_worker_data");
	add_topic("sensor_preflight_mag", 500);
	add_topic("actuator_test", 500);
	add_topic("neural_control", 50);
	add_topic("task_stack_info");
}

void LoggedTopics::add_estimator_replay_topics()
{
	// for estimator replay (need to be at full rate)
	add_topic("ekf2_timestamps");

	// current EKF2 subscriptions
	add_topic("airspeed");
	add_topic("airspeed_validated");
	add_topic("vehicle_optical_flow");
	add_topic("sensor_combined");
	add_topic("sensor_selection");
	add_topic("vehicle_air_data");
	add_topic("vehicle_gps_position");
	add_topic("vehicle_land_detected");
	add_topic("vehicle_magnetometer");
	add_topic("vehicle_status");
	add_topic("vehicle_visual_odometry");
	add_topic("ranging_beacon");
	add_topic_multi("aux_global_position");
	add_topic_multi("distance_sensor");
	add_topic("estimator_fusion_control", 200);

	// Every estimator* topic. This loop used to run in add_default_topics() - twice,
	// at 2 Hz and again at 100 Hz inside the SITL block - and between them they are
	// what exhausted MAX_TOPICS_NUM and pushed the *_groundtruth topics off the end.
	// EKF data is now opt-in: SDLOG_PROFILE +2 (ESTIMATOR_REPLAY).
	{
		const uint16_t kEKFIntervalMilliseconds = 10;	// 100 Hz

		const struct orb_metadata *const *topic_list = orb_get_topics();

		for (size_t i = 0; i < orb_topics_count(); i++) {
			if (strncmp(topic_list[i]->o_name, "estimator", 9) == 0) {
				add_optional_topic_multi(topic_list[i]->o_name, kEKFIntervalMilliseconds);
			}
		}
	}

	add_optional_topic("estimator_selector_status", 10);
	add_optional_topic_multi("yaw_estimator_status", 1000);
}

void LoggedTopics::add_thermal_calibration_topics()
{
	add_topic_multi("sensor_accel", 100, 4);
	add_topic_multi("sensor_baro", 100, 4);
	add_topic_multi("sensor_gyro", 100, 4);
	add_topic_multi("sensor_mag", 100, 4);
}

void LoggedTopics::add_sensor_comparison_topics()
{
	add_topic_multi("sensor_accel", 100, 4);
	add_topic_multi("sensor_baro", 100, 4);
	add_topic_multi("sensor_gyro", 100, 4);
	add_topic_multi("sensor_mag", 100, 4);

	// Moved out of add_default_topics(): the integrated IMU and the vibration and
	// clipping metrics. Nothing in this fork's analysis reads them, and they are
	// several subscriptions each. SDLOG_PROFILE +64 (SENSOR_COMPARISON) brings them back.
	add_topic_multi("vehicle_imu", 500, 4);
	add_topic_multi("vehicle_imu_status", 1000, 4);
	add_topic("sensors_status_imu", 200);
	add_topic("vehicle_air_data", 200);
	add_optional_topic_multi("vehicle_magnetometer", 500, 4);
	add_optional_topic("sensor_gyro_fft", 50);
	add_optional_topic("sensor_correction");
}

void LoggedTopics::add_vision_and_avoidance_topics()
{
	add_topic("collision_constraints");
	add_topic_multi("distance_sensor");
	add_topic("obstacle_distance_fused");
	add_topic("obstacle_distance");
	add_topic("vehicle_mocap_odometry", 30);
	add_topic("vehicle_visual_odometry", 30);
}

void LoggedTopics::add_raw_imu_gyro_fifo()
{
	add_topic("sensor_gyro_fifo");
}

void LoggedTopics::add_raw_imu_accel_fifo()
{
	add_topic("sensor_accel_fifo");
}

void LoggedTopics::add_system_identification_topics()
{
	// for system id need to log imu and controls at full rate
	add_topic("sensor_combined");
	add_topic("vehicle_angular_velocity");
	add_topic("vehicle_torque_setpoint");
	add_topic("vehicle_acceleration");
	add_topic("actuator_motors");
}

void LoggedTopics::add_high_rate_sensors_topics()
{
	add_topic_multi("distance_sensor", 10, 4);
	add_topic_multi("sensor_baro", 10, 4);
	add_topic_multi("sensor_optical_flow", 10, 2);
	add_topic_multi("sensor_gps", 10, 4);
	add_topic_multi("sensor_gnss_relative", 10, 1);
	add_topic_multi("sensor_mag", 10, 4);
	add_topic("estimator_aid_src_baro_hgt", 10);
	add_topic("vehicle_air_data", 10);
	add_topic("vehicle_magnetometer", 10);
	add_topic("vehicle_thrust_setpoint", 10);
}

void LoggedTopics::add_mavlink_tunnel()
{
	add_topic("mavlink_tunnel");
}

int LoggedTopics::add_topics_from_file(const char *fname)
{
	int ntopics = 0;

	/* open the topic list file */
	FILE *fp = fopen(fname, "r");

	if (fp == nullptr) {
		return -1;
	}

	/* call add_topic for each topic line in the file */
	for (;;) {
		/* get a line, bail on error/EOF */
		char line[80];
		line[0] = '\0';

		if (fgets(line, sizeof(line), fp) == nullptr) {
			break;
		}

		/* skip comment lines */
		if ((strlen(line) < 2) || (line[0] == '#')) {
			continue;
		}

		// read line with format: <topic_name>[ <interval>[ <instance>]]
		char topic_name[80];
		uint32_t interval_ms = 0;
		uint32_t instance = 0;
		int nfields = sscanf(line, "%s %" PRIu32 " %" PRIu32, topic_name, &interval_ms, &instance);

		if (nfields > 0) {
			int name_len = strlen(topic_name);

			if (name_len > 0 && topic_name[name_len - 1] == ',') {
				topic_name[name_len - 1] = '\0';
			}

			/* add topic with specified interval_ms */
			if ((nfields > 2 && add_topic(topic_name, interval_ms, instance))
			    || add_topic_multi(topic_name, interval_ms)) {
				ntopics++;

			} else {
				PX4_ERR("Failed to add topic %s", topic_name);
			}
		}
	}

	fclose(fp);
	return ntopics;
}

void LoggedTopics::initialize_mission_topics(MissionLogType mission_log_type)
{
	if (mission_log_type == MissionLogType::Complete) {
		add_mission_topic("camera_capture");
		add_mission_topic("mission_result");
		add_mission_topic("vehicle_global_position", 1000);
		add_mission_topic("vehicle_status", 1000);

	} else if (mission_log_type == MissionLogType::Geotagging) {
		add_mission_topic("camera_capture");
	}
}

void LoggedTopics::add_mission_topic(const char *name, uint16_t interval_ms)
{
	if (add_topic(name, interval_ms)) {
		++_num_mission_subs;
	}
}

bool LoggedTopics::add_topic(const orb_metadata *topic, uint16_t interval_ms, uint8_t instance, bool optional)
{
	if (_subscriptions.count >= MAX_TOPICS_NUM) {
		PX4_WARN("Too many subscriptions, failed to add: %s %" PRIu8, topic->o_name, instance);
		return false;
	}

	if (optional && orb_exists(topic, instance) != 0) {
		PX4_DEBUG("Not adding non-existing optional topic %s %i", topic->o_name, instance);

		if (instance == 0 && _subscriptions.num_excluded_optional_topic_ids < MAX_EXCLUDED_OPTIONAL_TOPICS_NUM) {
			_subscriptions.excluded_optional_topic_ids[_subscriptions.num_excluded_optional_topic_ids++] = topic->o_id;
		}

		return false;
	}

	RequestedSubscription &sub = _subscriptions.sub[_subscriptions.count++];
	sub.interval_ms = interval_ms;
	sub.instance = instance;
	sub.id = static_cast<ORB_ID>(topic->o_id);
	return true;
}

bool LoggedTopics::add_topic(const char *name, uint16_t interval_ms, uint8_t instance, bool optional)
{
	interval_ms /= _rate_factor;

	const orb_metadata *const *topics = orb_get_topics();
	bool success = false;

	for (size_t i = 0; i < orb_topics_count(); i++) {
		if (strcmp(name, topics[i]->o_name) == 0) {
			bool already_added = false;

			// check if already added: if so, only update the interval
			for (int j = 0; j < _subscriptions.count; ++j) {
				if (_subscriptions.sub[j].id == static_cast<ORB_ID>(topics[i]->o_id) &&
				    _subscriptions.sub[j].instance == instance) {

					PX4_DEBUG("logging topic %s(%" PRIu8 "), interval: %" PRIu16 ", already added, only setting interval",
						  topics[i]->o_name, instance, interval_ms);

					_subscriptions.sub[j].interval_ms = interval_ms;
					success = true;
					already_added = true;
					break;
				}
			}

			if (!already_added) {
				success = add_topic(topics[i], interval_ms, instance, optional);

				if (success) {
					PX4_DEBUG("logging topic: %s(%" PRIu8 "), interval: %" PRIu16, topics[i]->o_name, instance, interval_ms);
				}

				break;
			}
		}
	}

	return success;
}

bool LoggedTopics::add_topic_multi(const char *name, uint16_t interval_ms, uint8_t max_num_instances, bool optional)
{
	// add all possible instances
	for (uint8_t instance = 0; instance < max_num_instances; instance++) {
		add_topic(name, interval_ms, instance, optional);
	}

	return true;
}

bool LoggedTopics::initialize_logged_topics(SDLogProfileMask profile)
{
	// load appropriate topics for profile
	// the order matters: if several profiles add the same topic, the logging rate of the last one will be used
	if (profile & SDLogProfileMask::DEFAULT) {
		add_default_topics();
	}

	if (profile & SDLogProfileMask::ESTIMATOR_REPLAY) {
		add_estimator_replay_topics();
	}

	if (profile & SDLogProfileMask::THERMAL_CALIBRATION) {
		add_thermal_calibration_topics();
	}

	if (profile & SDLogProfileMask::SYSTEM_IDENTIFICATION) {
		add_system_identification_topics();
	}

	if (profile & SDLogProfileMask::HIGH_RATE) {
		add_high_rate_topics();
	}

	if (profile & SDLogProfileMask::DEBUG_TOPICS) {
		add_debug_topics();
	}

	if (profile & SDLogProfileMask::SENSOR_COMPARISON) {
		add_sensor_comparison_topics();
	}

	if (profile & SDLogProfileMask::VISION_AND_AVOIDANCE) {
		add_vision_and_avoidance_topics();
	}

	if (profile & SDLogProfileMask::RAW_IMU_GYRO_FIFO) {
		add_raw_imu_gyro_fifo();
	}

	if (profile & SDLogProfileMask::RAW_IMU_ACCEL_FIFO) {
		add_raw_imu_accel_fifo();
	}

	if (profile & SDLogProfileMask::MAVLINK_TUNNEL) {
		add_mavlink_tunnel();
	}

	if (profile & SDLogProfileMask::HIGH_RATE_SENSORS) {
		add_high_rate_sensors_topics();
	}

	int ntopics = add_topics_from_file(PX4_STORAGEDIR "/etc/logging/logger_topics.txt");

	if (ntopics > 0) {
		PX4_INFO("logging %d topics from logger_topics.txt", ntopics);

	} else if ((int32_t)profile == 0) {
		PX4_WARN("No logging topics added. Using default set");
		add_default_topics();
	}

	return _subscriptions.count > 0;
}
