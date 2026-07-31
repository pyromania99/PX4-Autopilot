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
 * @file ControllerIO.hpp
 * @brief Input and output types for pluggable multicopter control laws.
 *
 * FROZEN INTERFACE. Every controller and the framework itself depend on these
 * types; changing them is a breaking change for all controllers.
 *
 * Frame conventions used throughout:
 *   translational : NED earth-fixed, metres / m s^-1 / m s^-2
 *   rotational    : body FRD, rad s^-1 / rad s^-2
 *   q             : Hamilton quaternion rotating body-FRD -> NED (PX4 convention)
 *
 * NaN means "not estimable" in state and "not commanded" in setpoints.
 */

#pragma once

#include <matrix/matrix/math.hpp>

#include <cstdint>

namespace mc_ctrl
{

/**
 * Level at which the active flight mode hands over control.
 *
 * Resolved from vehicle_control_mode only; see CommandFrontEnd::resolveLevel().
 * Values match mc_controller_status_s::CONTROL_LEVEL_* (static_assert in the .cpp).
 */
enum class ControlLevel : uint8_t {
	None       = 0, ///< no control output (termination, disarmed, not rotary wing)
	BodyRate   = 1, ///< rate_sp + thrust_body_sp        (Acro, offboard body_rate)
	Attitude   = 2, ///< attitude_sp + thrust_body_sp    (Manual, Stabilized, offboard attitude)
	Trajectory = 3, ///< position/velocity/accel/yaw NED (Altitude, Position, Auto, Offboard, Orbit)
};

/// Bit for a level, for composing supportedLevels() masks.
static constexpr uint8_t levelBit(ControlLevel l) { return static_cast<uint8_t>(1u << static_cast<uint8_t>(l)); }

/// Mask for a controller that handles the whole stack.
static constexpr uint8_t kAllLevels = static_cast<uint8_t>(
		levelBit(ControlLevel::Trajectory) | levelBit(ControlLevel::Attitude) | levelBit(ControlLevel::BodyRate));

const char *levelName(ControlLevel level);

/**
 * How fresh each estimator source is on this cycle.
 *
 * The framework runs at gyro rate; position and attitude arrive slower. A
 * controller wanting stage-wise rate parity with the stock cascade must gate on
 * the *_new flags and use the matching dt.
 */
struct StateFreshness {
	float dt{0.f};          ///< [s] since the previous update(),         clamped [0.000125, 0.02]
	float dt_attitude{0.f}; ///< [s] since the previous vehicle_attitude, clamped [0.0002,   0.02]
	float dt_position{0.f}; ///< [s] since the previous local_position,   clamped [0.002,    0.04]
	bool attitude_new{false};
	bool position_new{false};
};

/// Estimator discontinuities since the previous update(). Zero/false when nothing happened.
struct EkfResets {
	matrix::Vector2f delta_xy{};	///< [m]     NED
	float delta_z{0.f};		///< [m]
	matrix::Vector2f delta_vxy{};	///< [m/s]   NED
	float delta_vz{0.f};		///< [m/s]
	float delta_heading{0.f};	///< [rad]
	matrix::Quatf delta_q{};	///< body-frame rotation delta

	bool xy{false};
	bool z{false};
	bool vxy{false};
	bool vz{false};
	bool heading{false};
	bool quat{false};

	void clear() { *this = EkfResets{}; }
	bool any() const { return xy || z || vxy || vz || heading || quat; }
};

/// Everything the estimator knows, plus the vehicle situation the control law needs.
struct ControllerState {
	uint64_t timestamp_sample{0};	///< [us] vehicle_angular_velocity.timestamp_sample

	matrix::Quatf q{};		///< always finite while the module runs
	matrix::Vector3f angular_velocity{};
	matrix::Vector3f angular_accel{};

	matrix::Vector3f position{};	///< NaN per axis when not estimable
	matrix::Vector3f velocity{};	///< filtered (MPC_VEL_NF_*, MPC_VEL_LP)
	matrix::Vector3f acceleration{};///< filtered derivative of velocity (MPC_VELD_LP)
	float heading{NAN};		///< [rad]
	float unaided_heading{NAN};	///< [rad], NaN when unavailable

	bool position_valid_xy{false};
	bool position_valid_z{false};
	bool velocity_valid_xy{false};
	bool velocity_valid_z{false};

	EkfResets resets{};
	StateFreshness freshness{};

	bool landed{true};
	bool maybe_landed{true};
	bool ground_contact{false};
	bool freefall{false};
	bool armed{false};
	bool spooled_up{false};		///< armed for longer than COM_SPOOLUP_TIME

	float hover_thrust{0.5f};	///< normalized, constrained [0.05, 0.9]
	bool hover_thrust_valid{false};	///< true when sourced from hover_thrust_estimate
};

/**
 * The commanded setpoint, at the level indicated by `level`.
 *
 * The framework guarantees:
 *  - BodyRate:   rate_sp is fully finite (NaN axes replaced with the measured rate)
 *  - Attitude:   attitude_sp is a finite unit quaternion and thrust_body_sp is finite
 *  - Trajectory: per-axis NaN semantics identical to trajectory_setpoint
 */
struct ControllerCommand {
	uint64_t timestamp{0};
	ControlLevel level{ControlLevel::None};

	// ---- Trajectory level (NED) ----
	matrix::Vector3f position_sp{NAN, NAN, NAN};
	matrix::Vector3f velocity_sp{NAN, NAN, NAN};
	matrix::Vector3f acceleration_sp{NAN, NAN, NAN};
	matrix::Vector3f jerk_sp{NAN, NAN, NAN};
	float yaw_sp{NAN};		///< [rad] absolute heading
	float yawspeed_sp{NAN};		///< [rad/s]

	// which axes the flight mode actually wants controlled
	bool axis_position{false};
	bool axis_velocity{false};
	bool axis_altitude{false};
	bool axis_climb_rate{false};
	bool axis_acceleration{false};

	// limits the framework computed and expects the controller to respect
	float vel_limit_xy{NAN};	///< [m/s]
	float vel_limit_up{NAN};	///< [m/s], includes the takeoff ramp
	float vel_limit_down{NAN};	///< [m/s]
	float thrust_min{0.f};		///< normalized collective
	float thrust_max{1.f};
	float tilt_limit{NAN};		///< [rad]

	// ---- Attitude level (body FRD) ----
	matrix::Quatf attitude_sp{};
	float yaw_sp_move_rate{0.f};	///< [rad/s] feed forward
	matrix::Vector3f thrust_body_sp{}; ///< normalized [-1,1]; MC: x=y=0, z<=0

	// ---- BodyRate level (body FRD) ----
	matrix::Vector3f rate_sp{};	///< [rad/s]

	// ---- context ----
	bool manual{false};
	bool automatic{false};
	bool offboard{false};

	/**
	 * True when the outer (trajectory -> attitude) stage already ran on the
	 * lower-priority work queue this cycle, so attitude_sp / thrust_body_sp /
	 * yaw_sp_move_rate are already populated and the controller must NOT re-run its
	 * own position stage. Only ever set at Trajectory level, and only for
	 * controllers whose hasOuterStage() returns true.
	 */
	bool outer_stage_complete{false};

	/**
	 * HARD CONTRACT: when true the controller MUST zero its integrators.
	 * Set on mode change, on the ground, and when disarmed. The framework cannot
	 * reach into a plugin's internal state, so this is the one behaviour that
	 * depends on controller cooperation.
	 */
	bool reset_integrals{false};
};

/**
 * What the control law produces.
 *
 * There is exactly ONE output contract: normalized torque and thrust, which
 * control_allocator mixes into actuator setpoints. A controller does not get to
 * publish actuator_motors itself - control_allocator publishes that topic
 * unconditionally on every cycle and is backup-scheduled at 20 Hz even with no
 * torque setpoint arriving, so a second publisher would simply contend with it.
 * Failure-tolerant or otherwise non-standard allocation belongs behind
 * CA_METHOD / the effectiveness matrix, not here.
 *
 * Defaults are deliberately invalid (valid == false, torque/thrust NaN) so a
 * controller that forgets to write its output fails closed rather than
 * commanding zero.
 */
struct ControllerOutput {
	/// Normalized [-1,1] body FRD.
	matrix::Vector3f torque{NAN, NAN, NAN};

	/**
	 * Normalized [-1,1] body FRD.
	 * REQUIRED: land_detector and mc_hover_thrust_estimator consume the published
	 * vehicle_thrust_setpoint, so omitting it silently breaks land detection and
	 * auto-land.
	 */
	matrix::Vector3f thrust{NAN, NAN, NAN};

	/// Optional diagnostics, republished by the framework for logging and consumers.
	matrix::Quatf attitude_setpoint{NAN, NAN, NAN, NAN};
	matrix::Vector3f rate_setpoint{NAN, NAN, NAN};

	/// Set true only when the controller trusts its own output this cycle.
	bool valid{false};

	void reset() { *this = ControllerOutput{}; }
};

/// Optional anti-windup feedback from control_allocator.
struct AllocatorFeedback {
	matrix::Vector3f unallocated_torque{};
	bool torque_setpoint_achieved{true};
	matrix::Vector<bool, 3> saturation_positive{};
	matrix::Vector<bool, 3> saturation_negative{};
};

/// Validate an output before it is allowed near the mixer.
bool outputIsFinite(const ControllerOutput &output);

/**
 * The framework copies ControllerState every gyro cycle, so guard against it
 * quietly growing into something expensive.
 */
static_assert(sizeof(ControllerState) <= 512, "ControllerState has grown unexpectedly large");

} // namespace mc_ctrl
