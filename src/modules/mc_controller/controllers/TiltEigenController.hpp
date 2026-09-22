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
 * @file TiltEigenController.hpp
 * @brief Eigen-dynamics controller on a tilt-axis (reduced attitude) error. MC_CTRL_ALG=4.
 *
 * The same control law as EigenController (MC_CTRL_ALG=3), with the Euler-angle attitude
 * chart replaced by a reduced-attitude error on S^2. Three stages:
 *
 *   1. outer   position/velocity PD                -> desired NED acceleration
 *   2. middle  desired acceleration                -> desired THRUST DIRECTION, and collective
 *   3. inner   feedback linearization placing the eigenvalue pair -wn +/- j*b on the
 *              roll/pitch subsystem                -> body torque [N m]
 *
 * WHY THIS EXISTS, given that MC_CTRL_ALG=3 is the same law. Stage 3 - the part that makes
 * either controller an "eigen" controller - is already chart-free: eigenTorque() consumes
 * body-frame rates and inertias and never mentions an Euler angle. The eigenvalue pair
 * -wn +/- j*b IS a rotation in the body x-y plane, and a tilt-axis error is a vector in
 * that same plane, so the cross-coupling ports across untouched. Euler was never load
 * bearing for the eigen structure; it was the accident of the Isaac Sim prototype. This
 * controller keeps the structure and drops the accident.
 *
 * WHAT CHANGES, concretely, relative to MC_CTRL_ALG=3:
 *
 *   - No gimbal-lock singularity. MC_CTRL_ALG=3 reads phi/theta out of an Euler
 *     decomposition and is singular at +/-90 deg of pitch; the reduced-attitude error is
 *     defined and finite at every attitude, including inverted.
 *
 *   - The small-angle inversion of R is gone. MC_CTRL_ALG=3 linearizes the hover relation,
 *     divides by g, then has to clamp the result a second time because the linearization
 *     overshoots the true angle (at a 45 deg limit it asks for 57 deg). Here the desired
 *     body-z is a direction, and limitTilt() bounds it exactly at any angle.
 *
 *   - Yaw decoupling is structural rather than constructed. MC_CTRL_ALG=3 injects
 *     state.heading to build the setpoint and then differences it back out, so "no yaw
 *     control" depends on that cancellation holding. A reduced-attitude error has no yaw
 *     component to begin with: it is an axis in the body x-y plane, by construction.
 *
 *   - THERE IS NO ANGLE-ERROR DERIVATIVE, and therefore no MC_TEIG_ATT_D. See the note on
 *     the missing D term below - this is the one deliberate behavioural difference, not
 *     just a change of chart.
 *
 * THE MISSING D TERM. MC_CTRL_ALG=3 finite-differences its Euler angle error across
 * attitude samples (MC_EIG_ATT_D). The tilt error is a vector in the BODY frame, and that
 * frame is itself rotating, so numerically differencing it picks up a spurious omega_z x e
 * term of magnitude r*|e|. At hover r is ~0 and it is invisible; under the rotor-failure
 * spin this law exists for, r is large and that term swamps the real error rate - the same
 * failure shape as the prototype's sign-inverted gyroscopic coupling, second order in body
 * rate and therefore invisible in gentle sweeps. Rather than correct it, the term is
 * removed: MC_TEIG_WN already acts on rate error, which IS derivative feedback on attitude,
 * so the angle D term was largely redundant with it. Dropping it also makes update() fully
 * independent of dt, which removes the seeded-first-sample and held-derivative machinery
 * MC_CTRL_ALG=3 needs.
 *
 * PHYSICAL UNITS, unchanged from MC_CTRL_ALG=3. The inner loop is a feedback linearization:
 * it computes a desired angular acceleration and multiplies by the inertia tensor, so its
 * natural output is N m. control_allocator wants a dimensionless setpoint, so
 * MC_TEIG_TRQ_MAX converts. MC_TEIG_IXX/IYY/IZZ are real kg m^2, not bare gains.
 *
 * YAW IS NOT CONTROLLED at Trajectory level, for the same reason as MC_CTRL_ALG=1 and 3: a
 * multirotor missing a rotor cannot hold heading, and surrendering yaw is what leaves
 * enough authority to hold position. RC yaw stick still works in Stabilized and Acro.
 *
 * GAIN COMPATIBILITY WITH MC_CTRL_ALG=3. The tilt error vector reduces to
 * (roll_error, pitch_error) to first order - equal sign and equal magnitude, pinned by
 * TiltEigenControllerTest.TiltErrorMatchesEulerErrorAtSmallAngles - so MC_TEIG_ATT_P
 * inherits MC_EIG_ATT_P's tuned value directly. Every other parameter feeds a stage that is
 * byte-for-byte the same computation. The defaults here are MC_EIG_*'s defaults.
 *
 * To add a controller see src/modules/mc_controller/README.md.
 */

#pragma once

#include "RateLimits.hpp"

#include <MulticopterControllerBase.hpp>

#include <uORB/topics/vehicle_local_position_setpoint.h>

class TiltEigenController : public MulticopterControllerBase
{
public:
	explicit TiltEigenController(ModuleParams *parent);
	~TiltEigenController() override = default;

	const char *name() const override { return "tilt_eigen"; }
	uint8_t supportedLevels() const override { return mc_ctrl::kAllLevels; }

	void reset() override;

	bool update(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
		    float dt, mc_ctrl::ControllerOutput &output) override;


	void fillLocalPositionSetpoint(vehicle_local_position_setpoint_s &sp) const override;

	void fillStatus(mc_controller_status_s &status) const override;

protected:
	void updateParams() override;

private:
	/**
	 * MC_ROLLRATE_MAX / MC_PITCHRATE_MAX / MC_YAWRATE_MAX on what the tilt stage asks for,
	 * where stock applies them. Not applied in Acro, whose setpoint is the pilot's own.
	 */
	RateLimits _rate_limits;

	/// Stages 1+2: position/velocity PD -> desired acceleration -> desired thrust
	/// DIRECTION and collective. Writes _body_z_setpoint, _attitude_setpoint,
	/// _thrust_setpoint.
	void stepTrajectoryToAttitude(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command);

	/// Reduced-attitude P law: cached desired thrust direction -> desired body rates.
	/// Runs every cycle at gyro rate against the current attitude, which is why the
	/// trajectory stage caches a DIRECTION rather than a rate.
	matrix::Vector3f tiltToRateSetpoint(const mc_ctrl::ControllerState &state, float yaw_rate_setpoint);

	/// Stage 3: desired body rates -> body torque [N m], before normalization.
	/// Identical to EigenController::eigenTorque(); it needed no change.
	matrix::Vector3f eigenTorque(const matrix::Vector3f &rate_setpoint, const matrix::Vector3f &angular_velocity) const;

	// Cached across cycles: the trajectory stage runs at position rate, the rest at gyro
	// rate. What crosses that boundary is a unit DIRECTION in NED, not an angle pair -
	// the whole point is that the error against it is rebuilt from the live attitude.
	matrix::Vector3f _body_z_setpoint{0.f, 0.f, 1.f};	///< NED, unit
	matrix::Quatf _attitude_setpoint{};
	matrix::Vector3f _thrust_setpoint{};
	matrix::Vector3f _rate_setpoint{};
	matrix::Vector3f _torque{};		///< [N m], before MC_TEIG_TRQ_MAX

	/// Telemetry only, for vehicle_local_position_setpoint.
	matrix::Vector3f _position_setpoint{NAN, NAN, NAN};
	matrix::Vector3f _velocity_setpoint{NAN, NAN, NAN};
	matrix::Vector3f _acceleration_setpoint{NAN, NAN, NAN};

	/// Last reduced-attitude error, body frame, for mc_controller_status.debug[]. The z
	/// component is structurally zero; only x and y are published, matching
	/// MC_CTRL_ALG=3's debug layout so the two are directly comparable in a log.
	matrix::Vector3f _tilt_error{};

	bool _position_stage_valid{false};

	// Cached parameters, so update() never touches the parameter system.
	matrix::Vector3f _pos_p{};	///< [1/s^2] (MC_TEIG_XY_P, MC_TEIG_XY_P, MC_TEIG_Z_P)
	matrix::Vector3f _pos_d{};	///< [1/s]   (MC_TEIG_XY_D, MC_TEIG_XY_D, MC_TEIG_Z_D)
	matrix::Vector3f _inertia{};	///< [kg m^2] (MC_TEIG_IXX, MC_TEIG_IYY, MC_TEIG_IZZ)
	float _att_p{0.f};		///< [1/s] MC_TEIG_ATT_P
	float _wn{0.f};			///< [rad/s] MC_TEIG_WN
	float _b{0.f};			///< [rad/s] MC_TEIG_B
	float _alpha{0.f};		///< [1/s] MC_TEIG_ALPHA
	float _beta{0.f};		///< [1/s] MC_TEIG_BETA
	float _inv_torque_max{1.f};	///< [1/(N m)] 1 / MC_TEIG_TRQ_MAX

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::MC_TEIG_XY_P>)    _param_mc_teig_xy_p,
		(ParamFloat<px4::params::MC_TEIG_XY_D>)    _param_mc_teig_xy_d,
		(ParamFloat<px4::params::MC_TEIG_Z_P>)     _param_mc_teig_z_p,
		(ParamFloat<px4::params::MC_TEIG_Z_D>)     _param_mc_teig_z_d,
		(ParamFloat<px4::params::MC_TEIG_ATT_P>)   _param_mc_teig_att_p,
		(ParamFloat<px4::params::MC_TEIG_WN>)      _param_mc_teig_wn,
		(ParamFloat<px4::params::MC_TEIG_B>)       _param_mc_teig_b,
		(ParamFloat<px4::params::MC_TEIG_ALPHA>)   _param_mc_teig_alpha,
		(ParamFloat<px4::params::MC_TEIG_BETA>)    _param_mc_teig_beta,
		(ParamFloat<px4::params::MC_TEIG_IXX>)     _param_mc_teig_ixx,
		(ParamFloat<px4::params::MC_TEIG_IYY>)     _param_mc_teig_iyy,
		(ParamFloat<px4::params::MC_TEIG_IZZ>)     _param_mc_teig_izz,
		(ParamFloat<px4::params::MC_TEIG_TRQ_MAX>) _param_mc_teig_trq_max
	)
};
