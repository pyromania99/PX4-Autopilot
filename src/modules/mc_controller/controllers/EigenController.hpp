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
 * @file EigenController.hpp
 * @brief Eigen-dynamics controller with a cross-coupled rate loop. MC_CTRL_ALG=3.
 *
 * Ported from the Isaac Sim prototype examples/utils/eigen_controller.py. Three stages:
 *
 *   1. outer   position/velocity PD                -> desired NED acceleration
 *   2. middle  desired acceleration + heading      -> desired roll/pitch, and collective
 *   3. inner   feedback linearization placing the eigenvalue pair -wn +/- j*b on the
 *              roll/pitch subsystem                -> body torque [N m]
 *
 * WHAT MAKES IT "EIGEN". Stage 3 does not treat roll and pitch as two independent axes.
 * A roll rate error produces pitch torque and vice versa, with opposite signs, so the
 * closed-loop roll/pitch subsystem has the complex-conjugate eigenvalues -wn +/- j*b and
 * is stabilized as one rotating mode. That is the mode a spinning or asymmetric vehicle
 * actually exhibits - which is the point on a vehicle that has lost a rotor.
 *
 * PHYSICAL UNITS. Unlike CascadedPdController, this law is a feedback linearization: it
 * computes a desired angular acceleration and multiplies by the inertia tensor, so its
 * natural output is N m. control_allocator wants a dimensionless setpoint
 * (ControlAllocationPseudoInverse::normalizeControlAllocationMatrix() divides the mix
 * columns by their own scale), so MC_EIG_TRQ_MAX converts. MC_EIG_IXX/IYY/IZZ are
 * therefore real kg m^2, not bare gains.
 *
 * YAW IS NOT CONTROLLED at Trajectory level, for the same reason as MC_CTRL_ALG=1: a
 * multirotor missing a rotor cannot hold heading, and surrendering yaw is what leaves
 * enough authority to hold position. Here it falls out of the algebra rather than being
 * imposed - see eigenTorque(). RC yaw stick still works in Stabilized and Acro.
 *
 * KNOWN LIMITATION: stages 2 and 3 are Euler-angle based, so this law has a gimbal-lock
 * singularity at +/-90 degrees of pitch that CascadedPdController's SO(3) formulation does
 * not. The tilt limit keeps normal flight far away from it; this is not an aerobatic
 * controller.
 *
 * To add a controller see src/modules/mc_controller/README.md.
 */

#pragma once

#include "Seat.hpp"
#include "PoleAdapter.hpp"

#include <MulticopterControllerBase.hpp>
#include <TrajectoryStage.hpp>

#include <uORB/topics/vehicle_local_position_setpoint.h>

class EigenController : public MulticopterControllerBase
{
public:
	explicit EigenController(ModuleParams *parent);
	~EigenController() override = default;

	const char *name() const override { return "eigen"; }
	uint8_t supportedLevels() const override { return mc_ctrl::kAllLevels; }

	void reset() override;

	bool update(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command,
		    float dt, mc_ctrl::ControllerOutput &output) override;


	void fillLocalPositionSetpoint(vehicle_local_position_setpoint_s &sp) const override;

	/// Freezes the seat's adaptation while the mixer is clipping roll or pitch.
	void setAllocatorFeedback(const mc_ctrl::AllocatorFeedback &feedback) override;

	void fillStatus(mc_controller_status_s &status) const override;

protected:
	void updateParams() override;

private:
	/// Stages 1+2-magnitude: position/velocity PD -> desired NED acceleration and collective.
	/// Writes _acceleration_setpoint and _thrust_setpoint. Stops short of the tilt setpoint
	/// on purpose - see _acceleration_setpoint.
	void stepTrajectoryToAcceleration(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command);

	/// Stage 2, direction: the cached NED acceleration resolved against the CURRENT heading
	/// into a small-angle roll/pitch pair. Writes _roll_setpoint, _pitch_setpoint and
	/// _attitude_setpoint.
	void resolveAccelerationToTilt(const mc_ctrl::ControllerState &state, const mc_ctrl::ControllerCommand &command);

	/// Angle PD: roll/pitch setpoint -> desired body rates. The derivative is a finite
	/// difference of the angle ERROR, refreshed only on a new attitude sample.
	matrix::Vector3f angleToRateSetpoint(const mc_ctrl::ControllerState &state, float yaw_rate_setpoint);

	/// Stage 3: desired body rates -> body torque [N m], before normalization.
	matrix::Vector3f eigenTorque(const matrix::Vector3f &rate_setpoint, const matrix::Vector3f &angular_velocity) const;

	// Cached across cycles: the trajectory stage runs at position rate, the angle-error
	// derivative at attitude rate, the rest at gyro rate.
	float _roll_setpoint{0.f};
	float _pitch_setpoint{0.f};
	matrix::Quatf _attitude_setpoint{};

	/**
	 * Desired NED acceleration [m/s^2], the heading-INDEPENDENT half of the tilt setpoint.
	 *
	 * The split exists so the two halves can run at their own rates. This vector is a
	 * function of the position estimate, so it is refreshed at position rate and held in
	 * between - correctly, since nothing about it has changed. Resolving it into roll and
	 * pitch requires the heading, which comes from the attitude estimate and arrives at
	 * gyro rate, so that resolution runs on EVERY cycle in update(). Holding the heading as
	 * well would leave the roll/pitch pair expressed in a frame the vehicle has rotated out
	 * of by r*dt_position, which the angle PD reads as a tilt error.
	 * See TrajectoryStage::currentHeading().
	 */
	matrix::Vector3f _acceleration_setpoint{};

	matrix::Vector3f _thrust_setpoint{};
	matrix::Vector3f _rate_setpoint{};
	matrix::Vector3f _torque{};		///< [N m], before MC_EIG_TRQ_MAX and before the seat

	/**
	 * Command-frame rotation of the roll/pitch torque, MC_SEAT_*. Off by default.
	 *
	 * Lives here rather than at the framework output stage because its observable needs
	 * THIS law's commanded angular acceleration and THIS law's assumed plant drift - the
	 * same A_r block eigenTorque() cancels. Nothing outside the controller has either.
	 * It is also where the level-1 twin sits, which is what makes level 2 minus level 1
	 * a measurement of PX4's cost rather than of two different laws.
	 */
	Seat _seat;

	/**
	 * Adaptive pole ANGLE (MC_POLE_*). Independent of the seat and off by default.
	 *
	 * Reads the same achieved-acceleration vector the seat's observable builds, but asks
	 * a different question of it: the seat asks "did the wrench arrive pointing where it
	 * was sent", this asks "is the acceleration the DESIGN asks for the one being
	 * achieved". It applies no rotation - theta_b takes effect through M_dyn on the next
	 * tick, by offsetting the pole angle the eigen block is built from.
	 */
	PoleAdapter _pole;

	// Angle-error derivative state. Held between attitude samples rather than
	// recomputed at gyro rate, where the difference would be quantization noise.
	float _roll_error_prev{0.f};
	float _pitch_error_prev{0.f};
	float _roll_error_rate{0.f};
	float _pitch_error_rate{0.f};
	bool _first_attitude_update{true};

	// Seat diagnostics, published through McControllerStatus::debug[5..7]. Kept here
	// rather than in Seat because xd and xa are built at the call site.
	float _wn_eff{0.f};		///< [rad/s] MC_EIG_WN rotated by the pole offset
	float _b_eff{0.f};		///< [rad/s] MC_EIG_B  rotated by the pole offset
	float _seat_saturated{0.f};
	float _seat_xd_norm{0.f};
	float _seat_xa_norm{0.f};
	float _seat_alpha_meas{NAN};	///< [rad] misalignment, measured on EVERY arm
	float _seat_stepped{NAN};	///< 1 = law stepped, 0 = gated, NAN = no law running

	/// Last roll/pitch angle error, for mc_controller_status.debug[].
	matrix::Vector2f _attitude_error{};

	/**
	 * Stages 1 and 2-magnitude, shared with the other laws in this module: the per-axis
	 * PD, the position integrator, the lateral clamp and the collective. Opting in is
	 * composition - holding one of these and delegating - so opting out would mean simply
	 * not having it, as TemplateController does.
	 *
	 * A ModuleParams child, so its own gains refresh through the same cascade this
	 * controller does. It runs inline in update() on the rate_ctrl queue, gated on
	 * state.freshness.position_new so it keeps position rate rather than gyro rate.
	 */
	TrajectoryStage _trajectory_stage;

	// Cached parameters, so update() never touches the parameter system.
	matrix::Vector3f _pos_p{};	///< [1/s^2] (MC_EIG_XY_P, MC_EIG_XY_P, MC_EIG_Z_P)
	matrix::Vector3f _pos_d{};	///< [1/s]   (MC_EIG_XY_D, MC_EIG_XY_D, MC_EIG_Z_D)
	matrix::Vector3f _inertia{};	///< [kg m^2] (MC_EIG_IXX, MC_EIG_IYY, MC_EIG_IZZ)
	float _att_p{0.f};		///< [1/s] MC_EIG_ATT_P
	float _att_d{0.f};		///< [-]   MC_EIG_ATT_D
	float _wn{0.f};			///< [rad/s] MC_EIG_WN
	float _b{0.f};			///< [rad/s] MC_EIG_B
	float _alpha{0.f};		///< [1/s] MC_EIG_ALPHA
	float _beta{0.f};		///< [1/s] MC_EIG_BETA
	float _inv_torque_max{1.f};	///< [1/(N m)] 1 / MC_EIG_TRQ_MAX

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::MC_EIG_XY_P>)    _param_mc_eig_xy_p,
		(ParamFloat<px4::params::MC_EIG_XY_D>)    _param_mc_eig_xy_d,
		(ParamFloat<px4::params::MC_EIG_Z_P>)     _param_mc_eig_z_p,
		(ParamFloat<px4::params::MC_EIG_Z_D>)     _param_mc_eig_z_d,
		(ParamFloat<px4::params::MC_EIG_ATT_P>)   _param_mc_eig_att_p,
		(ParamFloat<px4::params::MC_EIG_ATT_D>)   _param_mc_eig_att_d,
		(ParamFloat<px4::params::MC_EIG_WN>)      _param_mc_eig_wn,
		(ParamFloat<px4::params::MC_EIG_B>)       _param_mc_eig_b,
		(ParamFloat<px4::params::MC_EIG_ALPHA>)   _param_mc_eig_alpha,
		(ParamFloat<px4::params::MC_EIG_BETA>)    _param_mc_eig_beta,
		(ParamFloat<px4::params::MC_EIG_IXX>)     _param_mc_eig_ixx,
		(ParamFloat<px4::params::MC_EIG_IYY>)     _param_mc_eig_iyy,
		(ParamFloat<px4::params::MC_EIG_IZZ>)     _param_mc_eig_izz,
		(ParamFloat<px4::params::MC_EIG_TRQ_MAX>) _param_mc_eig_trq_max
	)
};
