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
 * @file TrajectoryStage.hpp
 * @brief The trajectory stage every control law here shares: setpoint -> NED
 *        acceleration, and acceleration -> collective.
 *
 * WHERE IT RUNS. Inline in the controller's update(), on the rate_ctrl work queue,
 * gated on state.freshness.position_new so it keeps position rate rather than gyro rate.
 * There is no second work queue and no cross-thread contract: the framework used to
 * carry an optional `OuterLoop` work item for a split trajectory stage, but no law ever
 * opted in and it was removed. Freshness gating is the cheap way to get stage-rate
 * parity, and it is rate-agnostic - the ratio follows whatever the gyro and position
 * rates happen to be.
 *
 * OPT-IN BY COMPOSITION. A law opts in by holding one of these and delegating; it opts
 * out by not holding one and writing its own stage, as TemplateController does. There is
 * no virtual and no runtime branch on the real-time path.
 *
 * WHAT IS SHARED, AND WHY ONLY THIS MUCH. Stage 1 (per-axis PD with NaN-aware setpoint
 * handling), the lateral acceleration clamp, the telemetry cache and the collective were
 * literally duplicated across CascadedPdController, EigenController and
 * TiltEigenController - byte for byte, modulo the parameter names backing the gains. What
 * is NOT shared is the acceleration -> attitude conversion, because that map is what
 * distinguishes the laws: one produces a quaternion via a body-z direction, one a pair of
 * small-angle Euler scalars, one a unit body-z vector. Those are three different
 * mathematical objects, not three spellings of one.
 *
 * GAINS ARE PASSED IN, not owned. Each law keeps its own MC_EIG_* / MC_PD_* values, whose
 * Z defaults genuinely differ. Only the integrator gains belong to this class, because
 * the integrator is new and has no per-law history to preserve.
 *
 * THE INTEGRATOR integrates POSITION error, not velocity error. That differs from stock
 * mc_pos_control deliberately: stock is a cascade whose position loop generates a velocity
 * setpoint, so a standing position offset shows up as a velocity error and integrating
 * velocity error corrects it. Here `pos_d` multiplies (command.velocity_sp -
 * state.velocity) directly, with no internally generated velocity setpoint, so at a steady
 * offset the velocity error is zero and integrating it would do nothing. Integrating
 * position error also self-disables on velocity-only axes, which is correct: ALTCTL
 * commands no XY position, so there is no XY hold target to trim.
 */

#pragma once

#include <ControllerIO.hpp>

#include <px4_platform_common/module_params.h>
#include <uORB/topics/vehicle_local_position_setpoint.h>

class TrajectoryStage : public ModuleParams
{
public:
	explicit TrajectoryStage(ModuleParams *parent);
	~TrajectoryStage() override = default;

	TrajectoryStage(const TrajectoryStage &) = delete;
	TrajectoryStage &operator=(const TrajectoryStage &) = delete;

	/**
	 * Stage 1: commanded setpoint -> desired NED acceleration.
	 *
	 * Per-axis PD on whatever the flight mode actually commanded, plus the position-error
	 * integral, bounded by what the tilt limit can deliver laterally. Caches the setpoints
	 * fillLocalPositionSetpoint() reports and marks the stage valid.
	 *
	 * @param pos_p [1/s^2] per-axis position gain, owned by the calling law
	 * @param pos_d [1/s]   per-axis velocity gain, owned by the calling law
	 */
	matrix::Vector3f computeAccelerationSetpoint(const mc_ctrl::ControllerState &state,
			const mc_ctrl::ControllerCommand &command,
			const matrix::Vector3f &pos_p,
			const matrix::Vector3f &pos_d);

	/**
	 * Stage 2, magnitude: desired NED acceleration -> body-FRD thrust setpoint (z <= 0).
	 *
	 * The vertical specific force the vehicle must produce is (g - a_z), and only
	 * T*cos(tilt) of the collective points up, so T = (g - a_z)/cos(tilt). Normalized by
	 * hover thrust - which is by definition what produces 1 g - that is
	 * h*(g - a_z)/(g*cos_tilt), i.e. the WHOLE expression divided by cos(tilt), not just
	 * the hover term.
	 *
	 * cos_tilt is used SIGNED and the division is skipped outside the valid cone. Taking
	 * fabsf() and flooring instead would boost the collective by up to 10x past ~84 deg of
	 * tilt and keep boosting while inverted - which commands maximum thrust sideways, and
	 * downward respectively. Outside the cone the honest answer is that the vertical
	 * channel is unavailable, so the demand is passed through undivided and the collective
	 * saturates low.
	 */
	matrix::Vector3f computeThrustSetpoint(const mc_ctrl::ControllerState &state,
			const mc_ctrl::ControllerCommand &command,
			const matrix::Vector3f &acceleration_sp) const;

	/**
	 * The heading a heading-referenced attitude setpoint must be anchored on, taken from
	 * the attitude quaternion rather than from state.heading.
	 *
	 * Both laws sharing this stage rebuild their attitude setpoint around the vehicle's
	 * CURRENT heading, so how fresh that heading is a control input, not telemetry.
	 * The two available sources are the same quantity - state.heading is
	 * Eulerf(q).psi() - and differ only in age:
	 *
	 *   state.heading   vehicle_local_position, published once per EKF2 fusion step
	 *                   (EKF2_PREDICT_US): 125 Hz at the defaults, 250 Hz at 4000 us
	 *   state.q         vehicle_attitude, which EKF2's output predictor strapdown-
	 *                   integrates from buffered IMU gyro at full IMU rate
	 *
	 * Anchoring on the slower one rotates the whole attitude setpoint frame r*dt_position
	 * behind the vehicle - 1.8 deg per cycle at 8 rad/s and 250 Hz, 3.7 deg at the EKF2
	 * defaults - and because that is a rotation of the reference frame rather than of the
	 * commanded tilt, it leaks into the roll and pitch error as a yaw-rate-proportional
	 * disturbance. It is invisible in hover and grows linearly with spin rate, which is
	 * exactly the regime these laws exist to survive.
	 *
	 * Falls back to state.heading and then to zero: the failsafe controller has to
	 * produce a finite output on the disarmed and NaN paths too.
	 */
	static float currentHeading(const mc_ctrl::ControllerState &state);

	/**
	 * Fill vehicle_local_position_setpoint from the cached stage outputs.
	 *
	 * @param attitude_sp     the law's attitude setpoint, needed to rotate thrust into NED
	 * @param thrust_setpoint the law's body-FRD thrust setpoint
	 */
	void fillLocalPositionSetpoint(vehicle_local_position_setpoint_s &sp,
				       const matrix::Quatf &attitude_sp,
				       const matrix::Vector3f &thrust_setpoint) const;

	/// Drop everything: integrator, cached setpoints, hover-thrust history, validity.
	void reset();

	/**
	 * Zero the integrator only. For command.reset_integrals, which the framework asserts
	 * on the ground, when disarmed and on every level change - so this is what keeps the
	 * integrator from winding up against the thrust clamp before takeoff.
	 */
	void resetIntegral();

	/// Force the next call to run the stage regardless of position freshness.
	void invalidateStage() { _stage_valid = false; }

	/// False until the stage has run once since the last reset(); backs the freshness gate.
	bool stageValid() const { return _stage_valid; }

	/// Integral term currently applied, [m/s^2] NED. Diagnostics only.
	const matrix::Vector3f &integral() const { return _integral; }

protected:
	void updateParams() override;

private:
	/// Advance the integrator, having already absorbed any hover-thrust change.
	void updateIntegral(const mc_ctrl::ControllerState &state, int axis, float position_error);

	/**
	 * Keep the collective continuous across a hover-thrust estimate update.
	 *
	 * The integral feeds acceleration_sp(2), which the collective scales by hover thrust,
	 * so the same integral means a different collective once the estimate moves. Solving
	 * T(h', a_z') == T(h, a_z) gives a_z' = g + (h/h')*(a_z - g); the difference goes into
	 * the integrator. cos_tilt cancels, so this holds for any attitude. Same derivation as
	 * stock PositionControl::setHoverThrust().
	 */
	void absorbHoverThrustChange(float hover_thrust);

	matrix::Vector3f _integral{};	///< [m/s^2] NED

	/// Telemetry only, for vehicle_local_position_setpoint.
	matrix::Vector3f _position_setpoint{NAN, NAN, NAN};
	matrix::Vector3f _velocity_setpoint{NAN, NAN, NAN};
	matrix::Vector3f _acceleration_setpoint{NAN, NAN, NAN};

	float _hover_thrust_prev{NAN};
	bool _stage_valid{false};

	// Cached parameters, so the stage never touches the parameter system at gyro rate.
	matrix::Vector3f _pos_i{};	///< [1/s^3] (MC_OL_XY_I, MC_OL_XY_I, MC_OL_Z_I)
	float _integral_limit{0.f};	///< [m/s^2] MC_OL_I_LIM

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::MC_OL_XY_I>)  _param_mc_ol_xy_i,
		(ParamFloat<px4::params::MC_OL_Z_I>)   _param_mc_ol_z_i,
		(ParamFloat<px4::params::MC_OL_I_LIM>) _param_mc_ol_i_lim
	)
};
